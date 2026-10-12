#!/usr/bin/env python3
"""
Evaluate a Qwen3 model (base or trained checkpoint) on MATH500 with vLLM and the
PRM800K grader. No parameter perturbation: this is plain accuracy.

Base models (name ends with -base, or --base-model) get a plain prompt and greedy
decoding; instruct/trained models get the chat template with non-thinking sampling.

Examples:
    # base model on the full 500
    python -m eval.eval_math500 --model-path /tmp/erin/Qwen3-0.6B-Base
    # trained checkpoint on the training split, greedy
    python -m eval.eval_math500 --model-path .../checkpoint-50 --model-name step50 \
        --data-path train_data/math500/train.jsonl --temperature 0 --max-new-tokens 2048
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rlvr_landscape.grading.grader import grade_answer


DEFAULT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within "
    r"\boxed{}."
)


# ============================================================
# Configuration
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None, help="Defaults to the last component of --model-path.")
    parser.add_argument(
        "--base-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Plain prompt, no chat template. Default: true if the model name ends with '-base'.",
    )
    parser.add_argument("--data-path", default="train_data/math500/test/test_500.jsonl")
    parser.add_argument("--output-dir", default="math500_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Sampling: defaults depend on base vs instruct (see below).
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    # vLLM
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of visible GPUs.")
    cfg = parser.parse_args()

    if cfg.model_name is None:
        cfg.model_name = Path(cfg.model_path.rstrip("/")).name
    if cfg.base_model is None:
        cfg.base_model = cfg.model_name.lower().endswith("-base")

    if cfg.base_model:
        # Base model: no Qwen3 chat template, greedy decoding. Base models may
        # not stop on <|im_end|>, so keep a bounded generation length.
        defaults = dict(enable_thinking=None, use_chat_template=False, temperature=0.0,
                        top_p=1.0, top_k=-1, max_new_tokens=8192, stop=None)
    else:
        # Qwen3 instruct model: official non-thinking sampling configuration.
        defaults = dict(enable_thinking=False, use_chat_template=True, temperature=0.7,
                        top_p=0.8, top_k=20, max_new_tokens=32768, stop=["<|im_end|>", "<|endoftext|>"])
    for key, value in defaults.items():
        if getattr(cfg, key, None) is None:
            setattr(cfg, key, value)

    stem = (
        f"{cfg.model_name.lower().replace('-', '_')}_max{cfg.max_new_tokens}"
        f"_temp{cfg.temperature}_{Path(cfg.data_path).stem}"
    )
    cfg.output_path = str(Path(cfg.output_dir) / f"{stem}_predictions.jsonl")
    cfg.summary_path = str(Path(cfg.output_dir) / f"{stem}_summary.json")
    return cfg


# ============================================================
# Text extraction helpers
# ============================================================

def extract_text_content(content: Any) -> str:

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            )
        ).strip()

    return str(content)


def extract_problem(
    record: dict[str, Any],
) -> str:

    if "problem" in record:
        return str(record["problem"]).strip()

    if (
        "prompt" in record
        and isinstance(record["prompt"], str)
    ):
        return record["prompt"].strip()

    if "messages" in record:

        parts = [
            extract_text_content(
                message.get("content", "")
            )
            for message in record["messages"]
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
            )
        ]

        return "\n".join(
            part
            for part in parts
            if part
        ).strip()

    raise KeyError(
        "record has no problem, prompt, or user message"
    )


def extract_braced_command(
    text: str,
    command: str,
) -> list[str]:

    results: list[str] = []

    pattern = re.compile(
        rf"\\{re.escape(command)}\s*\{{"
    )

    for match in pattern.finditer(text):

        start = match.end()

        depth = 1

        index = start

        while index < len(text) and depth:

            if text[index] == "{":
                depth += 1

            elif text[index] == "}":
                depth -= 1

            index += 1

        if depth == 0:

            results.append(
                text[
                    start:index - 1
                ].strip()
            )

    return results


# ============================================================
# Prediction extraction
# ============================================================

def extract_prediction(
    text: str,
) -> tuple[str, str]:
    """
    Extraction priority:

    1. \\boxed{...}
    2. <answer>...</answer>
    3. Final Answer: ...
    4. Answer: ...

    If none succeeds:

        ("", "no_answer_marker")
    """

    # --------------------------------------------------------
    # 1. \boxed{...}
    # --------------------------------------------------------

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    if boxed:

        answer = boxed[-1].strip()

        if answer:
            return answer, "boxed"

    # --------------------------------------------------------
    # 2. <answer>...</answer>
    # --------------------------------------------------------

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:

        answer = tags[-1].strip()

        if answer:
            return answer, "answer_tag"

    # --------------------------------------------------------
    # 3. Final Answer: ...
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            r"Final\s+Answer\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if matches:

        remainder = text[
            matches[-1].end():
        ].strip()

        for line in remainder.splitlines():

            line = line.strip()

            if line:
                return line, "final_answer"

    # --------------------------------------------------------
    # 4. Answer: ...
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            r"Answer\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if matches:

        remainder = text[
            matches[-1].end():
        ].strip()

        for line in remainder.splitlines():

            line = line.strip()

            if line:
                return line, "answer"

    # --------------------------------------------------------
    # No answer
    # --------------------------------------------------------

    return "", "no_answer_marker"


# ============================================================
# Gold answer
# ============================================================

def extract_gold(
    record: dict[str, Any],
) -> str:

    value = record.get(
        "answer",
        record.get("solution"),
    )

    if value is None:
        raise KeyError(
            "record has neither answer nor solution"
        )

    text = str(value).strip()

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:
        return tags[-1].strip()

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    if boxed:
        return boxed[-1]

    return text


# ============================================================
# Dataset loading
# ============================================================

def load_records(
    path: str,
    limit: int | None,
) -> list[dict[str, Any]]:

    data_path = Path(path)

    if data_path.suffix.lower() == ".jsonl":

        with data_path.open(
            encoding="utf-8"
        ) as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    else:

        with data_path.open(
            encoding="utf-8"
        ) as file:

            loaded = json.load(file)

        if isinstance(loaded, dict):

            loaded = loaded.get(
                "test",
                loaded.get(
                    "data",
                    loaded,
                ),
            )

        if not isinstance(
            loaded,
            list,
        ):

            raise TypeError(
                "JSON input must contain a list of examples"
            )

        records = loaded

    if limit is not None:
        records = records[:limit]

    return records


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt(
    tokenizer: Any,
    problem: str,
    cfg: argparse.Namespace,
) -> str:

    content = (
        problem
        + DEFAULT_SUFFIX
    )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    if cfg.base_model:

        return content

    # --------------------------------------------------------
    # Instruct model
    # --------------------------------------------------------

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": content,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    cfg = parse_args()

    # ========================================================
    # Seeds
    # ========================================================

    random.seed(cfg.seed)

    np.random.seed(cfg.seed)

    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


    # ========================================================
    # Print configuration
    # ========================================================

    print("=" * 70)

    print("Math500 Evaluation")

    print("=" * 70)

    print(
        f"Model:              {cfg.model_path}"
    )

    print(
        f"Dataset:            {cfg.data_path}"
    )

    print(
        f"Output:             {cfg.output_path}"
    )

    print()

    print("Model type:")

    if cfg.base_model:
        print("  Base model")
    else:
        print("  Instruct model")

    print()

    print("Prompt:")

    print(
        f"  chat_template:     "
        f"{cfg.use_chat_template}"
    )

    print(
        f"  enable_thinking:   "
        f"{cfg.enable_thinking}"
    )

    print()

    print("Sampling:")

    print(
        f"  temperature:       "
        f"{cfg.temperature}"
    )

    print(
        f"  top_p:             "
        f"{cfg.top_p}"
    )

    print(
        f"  top_k:             "
        f"{cfg.top_k}"
    )

    print(
        f"  max_new_tokens:    "
        f"{cfg.max_new_tokens}"
    )

    print()

    print("vLLM:")

    print(
        f"  dtype:             "
        f"{cfg.dtype}"
    )

    print(
        f"  max_model_len:     "
        f"{cfg.max_model_len}"
    )

    print(
        f"  gpu_memory_util:   "
        f"{cfg.gpu_memory_utilization}"
    )

    print(
        f"  tensor_parallel:   "
        f"{cfg.tensor_parallel_size}"
    )

    print("=" * 70)


    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ========================================================
    # vLLM
    # ========================================================

    llm = LLM(
        model=cfg.model_path,
        dtype=cfg.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        seed=cfg.seed,
    )


    # ========================================================
    # Sampling parameters
    # ========================================================

    sampling_kwargs = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "max_tokens": cfg.max_new_tokens,
        "seed": cfg.seed,
    }

    if cfg.stop is not None:
        sampling_kwargs["stop"] = cfg.stop

    sampling_params = SamplingParams(
        **sampling_kwargs
    )


    # ========================================================
    # Load dataset
    # ========================================================

    records = load_records(
        cfg.data_path,
        cfg.limit,
    )

    print()

    print(
        f"Number of examples: "
        f"{len(records)}"
    )


    # ========================================================
    # Prepare output
    # ========================================================

    output_path = Path(
        cfg.output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Prepare metrics
    # ========================================================

    correct = 0

    scored = 0

    num_truncated = 0

    num_length_finished = 0


    # ========================================================
    # Prepare prompts
    # ========================================================

    problems = [
        extract_problem(record)
        for record in records
    ]

    golds = [
        extract_gold(record)
        for record in records
    ]

    prompts = [
        format_prompt(
            tokenizer,
            problem,
            cfg,
        )
        for problem in problems
    ]


    # ========================================================
    # Generate
    # ========================================================

    print()

    print(
        "Starting generation..."
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    print(
        "Generation finished."
    )


    # ========================================================
    # Grade
    # ========================================================

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, (
            problem,
            gold,
            output,
        ) in enumerate(
            zip(
                problems,
                golds,
                outputs,
            )
        ):

            completion_output = (
                output.outputs[0]
            )

            completion = (
                completion_output.text
            )

            finish_reason = getattr(
                completion_output,
                "finish_reason",
                None,
            )

            if finish_reason == "length":

                num_length_finished += 1


            # =================================================
            # Extract answer
            # =================================================

            prediction, method = (
                extract_prediction(
                    completion
                )
            )


            # =================================================
            # Truncation
            # =================================================

            truncated = not bool(
                prediction
            )

            if truncated:

                num_truncated += 1

                reward = 0.0

                error = None

            else:

                try:

                    reward = float(
                        grade_answer(
                            given_answer=prediction,
                            ground_truth=gold,
                        )
                    )

                    error = None

                except Exception as exc:

                    reward = 0.0

                    error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )


            # =================================================
            # Metrics
            # =================================================

            correct += int(reward)

            scored += 1


            # =================================================
            # Save result
            # =================================================

            result = {

                "index": index,

                "problem": problem,

                "gold": gold,

                "completion": completion,

                "prediction": prediction,

                "extraction_method": method,

                "reward": reward,

                "truncated": truncated,

                "finish_reason": finish_reason,
            }

            if error is not None:

                result[
                    "grader_error"
                ] = error


            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


            # =================================================
            # Progress
            # =================================================

            if scored % 10 == 0:

                accuracy = (
                    correct / scored
                )

                truncation_rate = (
                    num_truncated
                    / scored
                )

                length_rate = (
                    num_length_finished
                    / scored
                )

                tqdm.write(
                    f"scored={scored} "
                    f"correct={correct} "
                    f"accuracy={accuracy:.4%} "
                    f"truncated={num_truncated} "
                    f"truncation_rate="
                    f"{truncation_rate:.4%} "
                    f"length_finish="
                    f"{num_length_finished} "
                    f"length_rate="
                    f"{length_rate:.4%}"
                )


    # ========================================================
    # Final metrics
    # ========================================================

    accuracy = (
        correct / scored
        if scored
        else float("nan")
    )

    truncation_rate = (
        num_truncated / scored
        if scored
        else float("nan")
    )

    length_finish_rate = (
        num_length_finished / scored
        if scored
        else float("nan")
    )


    # ========================================================
    # Summary
    # ========================================================

    summary_path = Path(
        cfg.summary_path
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "model_name": cfg.model_name,

        "model_path": cfg.model_path,

        "model_type": (
            "base"
            if cfg.base_model
            else "instruct"
        ),

        "data_path": cfg.data_path,

        "predictions_path": str(
            output_path.resolve()
        ),


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        "num_examples": scored,

        "num_correct": correct,

        "accuracy": accuracy,

        "accuracy_percent": (
            accuracy * 100
        ),

        "num_truncated": (
            num_truncated
        ),

        "truncation_rate": (
            truncation_rate
        ),

        "truncation_rate_percent": (
            truncation_rate * 100
        ),

        "num_length_finished": (
            num_length_finished
        ),

        "length_finish_rate": (
            length_finish_rate
        ),

        "length_finish_rate_percent": (
            length_finish_rate * 100
        ),


        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        "use_chat_template": (
            cfg.use_chat_template
        ),

        "enable_thinking": (
            cfg.enable_thinking
        ),


        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        "temperature": cfg.temperature,

        "top_p": cfg.top_p,

        "top_k": cfg.top_k,

        "max_new_tokens": (
            cfg.max_new_tokens
        ),

        "stop": cfg.stop,


        # ----------------------------------------------------
        # vLLM
        # ----------------------------------------------------

        "backend": "vllm",

        "dtype": cfg.dtype,

        "max_model_len": (
            cfg.max_model_len
        ),

        "gpu_memory_utilization": (
            cfg.gpu_memory_utilization
        ),

        "tensor_parallel_size": (
            cfg.tensor_parallel_size
        ),

        "seed": cfg.seed,


        # ----------------------------------------------------
        # Grader
        # ----------------------------------------------------

        "grader": (
            "swift.grading.grader.grade_answer"
        ),
    }


    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:

        json.dump(
            summary,
            summary_file,
            ensure_ascii=False,
            indent=2,
        )

        summary_file.write("\n")


    # ========================================================
    # Final output
    # ========================================================

    print()

    print("=" * 70)

    print(
        "Evaluation Finished"
    )

    print("=" * 70)

    print(
        f"Model:               "
        f"{cfg.model_path}"
    )

    print(
        f"Examples:            "
        f"{scored}"
    )

    print(
        f"Correct:             "
        f"{correct}"
    )

    print(
        f"Accuracy:            "
        f"{accuracy:.6f} "
        f"({accuracy:.2%})"
    )

    print(
        f"Truncated:           "
        f"{num_truncated}/{scored}"
    )

    print(
        f"Truncation rate:     "
        f"{truncation_rate:.6f} "
        f"({truncation_rate:.2%})"
    )

    print(
        f"Length finished:     "
        f"{num_length_finished}/{scored}"
    )

    print(
        f"Length finish rate:  "
        f"{length_finish_rate:.6f} "
        f"({length_finish_rate:.2%})"
    )

    print(
        f"Predictions:         "
        f"{output_path.resolve()}"
    )

    print(
        f"Summary:             "
        f"{summary_path.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
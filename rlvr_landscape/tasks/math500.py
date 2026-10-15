"""MATH500: prompt construction, answer extraction and reward."""

import re
from typing import Any

from rlvr_landscape.grading.grader import grade_answer
from rlvr_landscape.tasks.common import extract_prompt_from_messages


DEFAULT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within "
    r"\boxed{}."
)


def add_answer_instruction(
    problem: str,
) -> str:
    """
    Append DEFAULT_SUFFIX unless the problem already asks for \\boxed{}.

    The repo's train/test files already end with their own boxed instruction;
    the raw HF MATH500 file (test/test_500.jsonl) has none.
    """

    if "boxed" in problem:
        return problem

    return problem + DEFAULT_SUFFIX


def load_math500_example(
    example: dict[str, Any],
) -> dict[str, Any]:

    if "messages" not in example:

        raise KeyError(
            "Math500 example is missing 'messages'"
        )

    if "solution" not in example:

        raise KeyError(
            "Math500 example is missing 'solution'"
        )

    # Raw math problem, without suffix.
    prompt = extract_prompt_from_messages(
        example["messages"]
    )

    answer = str(
        example["solution"]
    ).strip()

    if not prompt:

        raise ValueError(
            "Math500 example has no user prompt"
        )

    if not answer:

        raise ValueError(
            "Math500 example has an empty solution"
        )

    return {
        "prompt": prompt,
        "answer": answer,
        "metadata": example.get(
            "metadata",
            {},
        ),
        "extra_info": example.get(
            "extra_info",
            {},
        ),
    }


def build_prompt_text(
    tokenizer,
    example,
    force_suffix: bool = False,
):
    """
    Same prompt as format_prompt() in eval/eval_math500.py
    and the training prompt in train/grpo_trl.py.

    force_suffix=True appends DEFAULT_SUFFIX even when the problem already
    has its own boxed instruction (the prompt used before 4c077dc).
    """

    if force_suffix:
        content = example["prompt"] + DEFAULT_SUFFIX
    else:
        content = add_answer_instruction(
            example["prompt"]
        )

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


def extract_answer_tag(
    text: str,
) -> tuple[str, bool]:
    """Extract the final <answer>...</answer> block."""
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if matches:
        return matches[-1].strip(), True

    return str(text).strip(), False


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


def extract_math500_answer(
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

    text = str(text)

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

    return "", "no_answer_marker"


def extract_math500_gold(
    answer: Any,
) -> str:
    """
    EXACTLY aligned with eval_math500_displacement.py:

        <answer>...</answer>
        -> \\boxed{...}
        -> raw text
    """

    if answer is None:
        return ""

    if isinstance(answer, dict):

        if "answer" in answer:
            answer = answer["answer"]

        elif "solution" in answer:
            answer = answer["solution"]

    text = str(answer).strip()

    # --------------------------------------------------------
    # 1. <answer>...</answer>
    # --------------------------------------------------------

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:
        return tags[-1].strip()

    # --------------------------------------------------------
    # 2. \boxed{...}
    # --------------------------------------------------------

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    if boxed:
        return boxed[-1].strip()

    # --------------------------------------------------------
    # 3. raw text
    # --------------------------------------------------------

    return text


def math500_reward_func(
    completion: str,
    answer: Any,
    **_: Any,
) -> float:
    """
    EXACTLY aligned with eval_math500_displacement.py.

    Empty / unextractable prediction -> reward 0.

    Otherwise:

        grade_answer(
            given_answer=prediction,
            ground_truth=gold,
        )
    """

    prediction, method = (
        extract_math500_answer(
            completion
        )
    )

    gold = extract_math500_gold(
        answer
    )

    # Same behavior as displacement evaluator:
    # no answer marker -> reward = 0
    if not prediction:
        return 0.0

    if not gold:
        return 0.0

    try:

        reward = float(
            grade_answer(
                given_answer=prediction,
                ground_truth=gold,
            )
        )

        return reward

    except Exception as error:

        print(
            "[math500_reward_func] "
            f"prediction={prediction!r}, "
            f"gold={gold!r}, "
            f"method={method}, "
            f"error={type(error).__name__}: "
            f"{error}",
            flush=True,
        )

        return 0.0

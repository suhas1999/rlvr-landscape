"""GSM8K: example loading, numeric answer extraction and reward."""

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from rlvr_landscape.tasks.common import extract_prompt_from_messages


def load_gsm8k_example(
    example: dict[str, Any],
) -> dict[str, Any]:

    if "messages" not in example:
        raise KeyError(
            "GSM8K example is missing 'messages'"
        )

    if "solution" not in example:
        raise KeyError(
            "GSM8K example is missing 'solution'"
        )

    prompt = extract_prompt_from_messages(
        example["messages"]
    )

    if not prompt:
        raise ValueError(
            "GSM8K example has no user prompt"
        )

    return {
        "prompt": prompt,
        "answer": str(
            example["solution"]
        ).strip(),
        "metadata": example.get(
            "metadata",
            {},
        ),
        "extra_info": example.get(
            "extra_info",
            {},
        ),
    }


NUMBER_PATTERN = (
    r"[-+]?"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+"
    r"|"
    r"\d+(?:\.\d*)?"
    r"|"
    r"\.\d+"
    r")"
    r"(?:[eE][-+]?\d+)?"
)


def normalize_number(
    text: str,
) -> str:

    return (
        str(text)
        .replace(",", "")
        .replace("$", "")
        .strip()
        .rstrip(".")
    )


def parse_decimal(
    text: str,
) -> Decimal | None:

    cleaned = normalize_number(text)

    if not cleaned:
        return None

    try:

        return Decimal(cleaned)

    except InvalidOperation:

        return None


def extract_last_number(
    text: str,
) -> str:

    matches = list(
        re.finditer(
            NUMBER_PATTERN,
            text,
        )
    )

    if not matches:
        return ""

    return normalize_number(
        matches[-1].group(0)
    )


def extract_final_answer(
    text: str,
) -> str:

    boxed_matches = re.findall(
        r"\\boxed\s*\{\s*([^{}]+?)\s*\}",
        text,
    )

    if boxed_matches:

        extracted = extract_last_number(
            boxed_matches[-1]
        )

        if extracted:
            return extracted

    final_answer_match = re.search(
        r"Final\s+Answer\s*:?\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if final_answer_match:

        extracted = extract_last_number(
            final_answer_match.group(1)
        )

        if extracted:
            return extracted

    return ""


def extract_ground_truth(
    answer: str,
) -> str:

    answer = str(answer).strip()

    if "####" in answer:

        final_part = answer.rsplit(
            "####",
            1,
        )[-1]

        extracted = extract_last_number(
            final_part
        )

        if extracted:
            return extracted

    if re.fullmatch(
        NUMBER_PATTERN,
        answer,
    ):
        return normalize_number(answer)

    return extract_final_answer(answer)


def gsm8k_reward_func(
    completion: str,
    answer: str,
    **_: Any,
) -> float:

    prediction_text = extract_final_answer(
        completion
    )

    target_text = extract_ground_truth(
        answer
    )

    prediction = parse_decimal(
        prediction_text
    )

    target = parse_decimal(
        target_text
    )

    if prediction is None or target is None:
        return 0.0

    return float(
        prediction == target
    )

"""Chess single-turn move prediction: example loading, move parsing and reward."""

import ast
import json
import os
import re
from typing import Any


def load_chess_example(
    example: dict[str, Any],
) -> dict[str, Any]:

    if "prompt" not in example:
        raise KeyError(
            "Chess example is missing 'prompt'"
        )

    reward_model = example.get(
        "reward_model",
        {},
    )

    if "ground_truth" not in reward_model:
        raise KeyError(
            "Chess example is missing "
            "'reward_model.ground_truth'"
        )

    extra_info = example.get(
        "extra_info",
        {},
    )

    metadata = dict(
        example.get(
            "metadata",
            {},
        )
    )

    metadata.setdefault(
        "data_source",
        example.get("data_source"),
    )

    metadata.setdefault(
        "ability",
        example.get("ability"),
    )

    metadata.setdefault(
        "difficulty",
        example.get("difficulty"),
    )

    for key in (
        "FEN",
        "Moves",
        "PuzzleId",
        "Rating",
        "Themes",
        "env_replies",
        "first_move_san",
        "first_move_uci",
        "second_move_uci",
        "original_FEN",
    ):

        if key in extra_info:
            metadata.setdefault(
                key,
                extra_info[key],
            )

    return {
        "prompt": str(
            example["prompt"]
        ).strip(),

        "answer": str(
            reward_model["ground_truth"]
        ).strip(),

        "metadata": metadata,
        "extra_info": extra_info,
    }


CHESS_SINGLE_TURN_REWARD_MODEL_TYPE = (
    os.environ.get(
        "REWARD_MODEL_TYPE",
        "RULE_BASED",
    ).upper()
)


def get_fen(
    metadata: dict[str, Any],
    extra_info: dict[str, Any],
) -> str:

    for source in (
        metadata,
        extra_info,
    ):

        fen = (
            source.get("FEN")
            or source.get("fen")
        )

        if fen:
            return str(fen)

    return ""


def lan_to_uci(
    lan: str,
    side_to_move: str = "white",
) -> str:

    lan = lan.rstrip(
        "+#"
    ).strip()

    if lan == "O-O":

        if side_to_move == "white":
            return "e1g1"

        return "e8g8"

    if lan == "O-O-O":

        if side_to_move == "white":
            return "e1c1"

        return "e8c8"

    match = re.match(
        r"^([PNBRQK])"
        r"([a-h][1-8])"
        r"(x)?"
        r"([a-h][1-8])"
        r"(=([QRBN]))?$",
        lan,
    )

    if not match:
        raise ValueError(
            f"Invalid LAN format: {lan}"
        )

    (
        _piece,
        from_square,
        _capture,
        to_square,
        _promo_group,
        promo,
    ) = match.groups()

    uci = (
        from_square
        + to_square
    )

    if promo:
        uci += promo.lower()

    return uci


def is_complete_move(
    text: str,
) -> bool:

    if not text:
        return False

    move = text.rstrip(
        "+#"
    )

    if move in (
        "O-O",
        "O-O-O",
    ):
        return True

    return bool(
        re.match(
            r"^[PNBRQK]"
            r"[a-h][1-8]"
            r"(x)?"
            r"[a-h][1-8]"
            r"(=[QRBN])?$",
            move,
        )
    )


def extract_first_move(
    text: str,
) -> str | None:

    tokens = text.strip().split()

    for token in tokens:

        if re.match(
            r"^\d+\.{1,3}$",
            token,
        ):
            continue

        if is_complete_move(token):
            return token

    return None


def extract_move_after_thinking(
    text: str,
) -> tuple[str | None, bool]:

    text = text.strip()

    follows_format = (
        text.count("</T>") == 1
    )

    if not follows_format:
        return None, False

    text_after_thinking = text[
        text.find("</T>")
        + len("</T>"):
    ].strip()

    if not text_after_thinking:
        return None, True

    return (
        extract_first_move(
            text_after_thinking
        ),
        True,
    )


def parse_chess_ground_truth(
    answer: Any,
) -> str:

    ground_truth = answer

    if isinstance(
        ground_truth,
        str,
    ):

        try:

            ground_truth = json.loads(
                ground_truth
            )

        except json.JSONDecodeError:

            try:

                ground_truth = ast.literal_eval(
                    ground_truth
                )

            except (
                ValueError,
                SyntaxError,
            ):
                pass

    if (
        isinstance(
            ground_truth,
            list,
        )
        and ground_truth
    ):
        return str(
            ground_truth[0]
        ).strip()

    return str(
        ground_truth
    ).strip()


def chess_single_turn_move_to_uci(
    move_text: str,
) -> str:

    if not move_text:
        return ""

    try:

        return lan_to_uci(
            move_text
        )

    except ValueError:

        return ""


def extract_chess_single_turn_move(
    completion: str,
) -> tuple[str, bool]:

    move, follows_format = (
        extract_move_after_thinking(
            completion
        )
    )

    if (
        move is None
        and not follows_format
    ):
        move = extract_first_move(
            completion
        )

    return (
        move.strip()
        if move
        else "",
        follows_format,
    )


def chess_single_turn_reward_func(
    completion: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:

    raw_move, follows_format = (
        extract_chess_single_turn_move(
            completion
        )
    )

    prediction = (
        chess_single_turn_move_to_uci(
            raw_move
        )
    )

    target = parse_chess_ground_truth(
        answer
    )

    score = float(
        bool(prediction)
        and prediction == target
    )

    if (
        CHESS_SINGLE_TURN_REWARD_MODEL_TYPE
        == "RULE_FORMAT_BASED"
        and not follows_format
    ):
        return 0.0

    return score

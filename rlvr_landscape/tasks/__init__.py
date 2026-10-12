"""Task registry: dataset loading and reward-function selection."""

import json
from typing import Any

from rlvr_landscape.tasks.chess_single_turn import chess_single_turn_reward_func, load_chess_example
from rlvr_landscape.tasks.common import RewardFn
from rlvr_landscape.tasks.gsm8k import gsm8k_reward_func, load_gsm8k_example
from rlvr_landscape.tasks.math500 import load_math500_example, math500_reward_func


def load_eval_data(
    path: str,
    limit: int,
    task: str,
) -> list[dict[str, Any]]:

    if limit <= 0:
        raise ValueError(
            f"limit must be positive, got {limit}"
        )

    data: list[dict[str, Any]] = []

    with open(
        path,
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            try:

                example = json.loads(line)

                if task == "gsm8k":

                    parsed_example = (
                        load_gsm8k_example(
                            example
                        )
                    )

                elif task == "math500":

                    parsed_example = (
                        load_math500_example(
                            example
                        )
                    )

                else:

                    parsed_example = (
                        load_chess_example(
                            example
                        )
                    )

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:

                raise ValueError(
                    f"Failed to parse {task} "
                    f"example at {path}:{line_number}: "
                    f"{error}"
                ) from error

            data.append(
                parsed_example
            )

            if len(data) >= limit:
                break

    if not data:
        raise ValueError(
            f"No evaluation examples loaded from {path}"
        )

    return data


def select_reward_func(
    task: str,
) -> RewardFn:

    if task == "chess":
        return chess_single_turn_reward_func

    if task == "math500":
        return math500_reward_func

    return gsm8k_reward_func

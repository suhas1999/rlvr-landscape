"""Helpers shared by all tasks for reading chat-formatted dataset records."""

from typing import Any, Callable

RewardFn = Callable[..., float]


def extract_text_content(
    content: Any,
) -> str:

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


def extract_prompt_from_messages(
    messages: list[dict[str, Any]],
) -> str:
    """
    Extract raw user problem only.

    IMPORTANT:
    DEFAULT_SUFFIX is NOT added here.
    """

    prompt_parts = []

    for message in messages:

        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        text = extract_text_content(
            message.get(
                "content",
                "",
            )
        )

        if text:
            prompt_parts.append(text)

    return "\n".join(
        prompt_parts
    ).strip()

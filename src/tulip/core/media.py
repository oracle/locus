# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Images inside tool results.

A tool result is a string everywhere in Tulip: in :class:`~tulip.core.messages.ToolResult`,
in checkpoints, in the audit trail and across a durable engine's activity
boundary. A tool that returns a screenshot therefore embeds it in that string::

    return f"clicked; the page is at {url}" + encode_image(png)

Adapters that can show a model an image (Anthropic, OpenAI's Responses API)
split the string with :func:`split_content` and send real image parts; the rest
see :func:`strip_images`. Tulip's own bookkeeping measures a result with
:func:`text_length`, so a screenshot is neither truncated into corrupt base64
nor counted as thousands of text tokens.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass


_OPEN = "[tulip-image media_type="
_CLOSE = "[/tulip-image]"
_PATTERN = re.compile(
    r"\n?\[tulip-image media_type=(?P<media_type>[\w.+-]+/[\w.+-]+)\]\n"
    r"(?P<data>[A-Za-z0-9+/=\s]*?)\n\[/tulip-image\]\n?"
)

#: Text an adapter shows in place of an image it does not send.
IMAGE_OMITTED = "[image omitted]"

#: Tokens a model typically spends on one screenshot, for context estimates.
IMAGE_TOKEN_ESTIMATE = 1_600

#: How many of the latest image-bearing tool results an adapter sends as images.
#: Older screenshots are stale and expensive; they become a text placeholder.
RECENT_IMAGE_RESULTS = 3

#: The placeholder for a screenshot older than :data:`RECENT_IMAGE_RESULTS`.
EARLIER_IMAGE_OMITTED = "[earlier screenshot omitted]"


@dataclass(frozen=True)
class ImagePart:
    """One embedded image."""

    media_type: str
    data: str
    """Base64-encoded bytes."""

    @property
    def data_url(self) -> str:
        """The image as a ``data:`` URL."""
        return f"data:{self.media_type};base64,{self.data}"


def encode_image(data: bytes, media_type: str = "image/png") -> str:
    """``data`` as a segment to append to a tool result string."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"\n{_OPEN}{media_type}]\n{encoded}\n{_CLOSE}\n"


def has_images(content: str | None) -> bool:
    """Whether ``content`` embeds at least one image."""
    return bool(content) and _OPEN in content  # type: ignore[operator]


def split_content(content: str) -> list[str | ImagePart]:
    """``content`` as text and image parts, in order; empty text is dropped.

    A segment whose payload is not valid base64 stays text, so a model that
    echoes the marker cannot make an adapter send garbage as an image.
    """
    parts: list[str | ImagePart] = []
    position = 0
    for match in _PATTERN.finditer(content):
        data = re.sub(r"\s+", "", match.group("data"))
        try:
            base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            continue
        if match.start() > position:
            parts.append(content[position : match.start()])
        parts.append(ImagePart(media_type=match.group("media_type"), data=data))
        position = match.end()
    if position < len(content):
        parts.append(content[position:])
    return [part for part in parts if not isinstance(part, str) or part.strip()]


def images(content: str | None) -> list[ImagePart]:
    """The images embedded in ``content``."""
    if not has_images(content):
        return []
    return [part for part in split_content(content or "") if isinstance(part, ImagePart)]


def strip_images(content: str, placeholder: str = IMAGE_OMITTED) -> str:
    """``content`` with each image replaced by ``placeholder``."""
    if not has_images(content):
        return content
    return "\n".join(
        part if isinstance(part, str) else placeholder for part in split_content(content)
    )


def text_length(content: str | None) -> int:
    """Characters of text in ``content``, not counting embedded images."""
    if not content:
        return 0
    if not has_images(content):
        return len(content)
    return sum(len(part) for part in split_content(content) if isinstance(part, str))


def recent_image_positions(
    contents: list[str | None], keep: int = RECENT_IMAGE_RESULTS
) -> set[int]:
    """Positions in ``contents`` of the last ``keep`` entries that embed images."""
    with_images = [index for index, content in enumerate(contents) if has_images(content)]
    return set(with_images[-keep:]) if keep > 0 else set()


def estimate_tokens(content: str | None) -> int:
    """A rough token count: four characters of text per token, plus each image."""
    return text_length(content) // 4 + IMAGE_TOKEN_ESTIMATE * len(images(content))


__all__ = [
    "EARLIER_IMAGE_OMITTED",
    "IMAGE_OMITTED",
    "IMAGE_TOKEN_ESTIMATE",
    "RECENT_IMAGE_RESULTS",
    "ImagePart",
    "encode_image",
    "estimate_tokens",
    "has_images",
    "images",
    "recent_image_positions",
    "split_content",
    "strip_images",
    "text_length",
]

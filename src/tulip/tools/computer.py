# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Computer use: a model drives a browser by screenshot, pointer and keyboard.

:func:`computer_tool` is one ordinary :class:`~tulip.tools.decorator.Tool` named
``computer``. On Anthropic it is declared as Claude's native computer tool
(``computer_20251124``), on OpenAI's Responses API as the native ``computer``
tool; any other adapter sees an equivalent function schema. Whatever the
provider, each action the model asks for is performed on a
:class:`~tulip.tools.browser.BrowserSession` page and the result carries a fresh
screenshot.

Because it is a plain tool, hooks, ``gate_tool`` and the audit trail see every
action. :func:`computer_action` describes a call for policy: reads
(``screenshot``, ``wait``, pointer moves) carry ``computer-read``, anything that
clicks, types, drags or scrolls carries ``computer-write``, and a call OpenAI
flagged with a pending safety check carries ``computer-safety-check``::

    session = BrowserSession(allowed_domains=["shop.example.com"])
    computer = gate_tool(
        computer_tool(
            session,
            start_url="https://shop.example.com/orders",
            on_safety_check="proceed",
        ),
        policy=ControlPolicy(
            require_human_for=frozenset({"computer-safety-check"})
        ),
        action=computer_action,
        approval=store,
        on_refusal="interrupt",
    )

OpenAI's safety checks: a call that arrives with pending checks is refused
unless ``on_safety_check="proceed"``; set that only when the call is gated as
above, since running it acknowledges the checks to OpenAI. After every action
that can navigate, the page is checked against the session's
``allowed_domains``. Needs Playwright (``pip install "tulip-agents[browser]"``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from tulip.core.media import encode_image
from tulip.security.policy import Action
from tulip.tools.decorator import Tool


if TYPE_CHECKING:
    from tulip.tools.browser import BrowserSession


READ_LABELS = frozenset({"computer", "computer-read"})
WRITE_LABELS = frozenset({"computer", "computer-write"})
SAFETY_CHECK_LABEL = "computer-safety-check"

#: Anthropic computer tool versions and the beta header each needs.
ANTHROPIC_TOOL_BETAS = {
    "computer_20251124": "computer-use-2025-11-24",
    "computer_20250124": "computer-use-2025-01-24",
}

#: Actions that only look (Anthropic and OpenAI names).
READ_ACTIONS = frozenset({"screenshot", "wait", "cursor_position", "mouse_move", "move"})

_WHEEL_PIXELS = 100
_MAX_WAIT_SECONDS = 10.0

_KEYS = {
    "return": "Enter",
    "enter": "Enter",
    "ctrl": "Control",
    "control": "Control",
    "alt": "Alt",
    "option": "Alt",
    "shift": "Shift",
    "super": "Meta",
    "cmd": "Meta",
    "command": "Meta",
    "meta": "Meta",
    "win": "Meta",
    "esc": "Escape",
    "escape": "Escape",
    "backspace": "Backspace",
    "tab": "Tab",
    "space": "Space",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "page_up": "PageUp",
    "pageup": "PageUp",
    "prior": "PageUp",
    "page_down": "PageDown",
    "pagedown": "PageDown",
    "next": "PageDown",
    "up": "ArrowUp",
    "arrowup": "ArrowUp",
    "down": "ArrowDown",
    "arrowdown": "ArrowDown",
    "left": "ArrowLeft",
    "arrowleft": "ArrowLeft",
    "right": "ArrowRight",
    "arrowright": "ArrowRight",
}
_MODIFIERS = frozenset({"Control", "Alt", "Shift", "Meta"})

_DESCRIPTION = (
    "Use the browser through its screen: take a screenshot, move and click the "
    "pointer at pixel coordinates, type, press keys, drag and scroll. Every action "
    "returns a new screenshot."
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "screenshot",
                "left_click",
                "right_click",
                "middle_click",
                "double_click",
                "triple_click",
                "left_click_drag",
                "left_mouse_down",
                "left_mouse_up",
                "mouse_move",
                "type",
                "key",
                "hold_key",
                "scroll",
                "wait",
                "cursor_position",
            ],
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "[x, y] in screenshot pixels",
        },
        "start_coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "[x, y] where a drag starts",
        },
        "text": {"type": "string", "description": "Text to type, or keys like ctrl+a"},
        "scroll_direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
        "scroll_amount": {"type": "integer", "description": "Wheel clicks to scroll"},
        "duration": {"type": "number", "description": "Seconds, for wait and hold_key"},
    },
    "required": ["action"],
}


class ComputerError(Exception):
    """The computer tool refused or could not perform an action."""


def computer_action(name: str, args: dict[str, Any]) -> Action:
    """Describe a computer call for policy, as ``gate_tool(action=...)`` expects.

    ``asset`` names the actions; ``tags`` carry ``computer-read`` or
    ``computer-write`` and, for a call with pending safety checks,
    ``computer-safety-check``.
    """
    steps = _steps(args)
    kinds = [kind for kind, _ in steps]
    tags = set(READ_LABELS if all(kind in READ_ACTIONS for kind in kinds) else WRITE_LABELS)
    if args.get("pending_safety_checks"):
        tags.add(SAFETY_CHECK_LABEL)
    return Action(name=name, asset=",".join(kinds), kind="computer", tags=frozenset(tags))


def computer_tool(
    session: BrowserSession,
    *,
    display_width: int = 1280,
    display_height: int = 800,
    start_url: str | None = None,
    anthropic_version: str = "computer_20251124",
    openai_tool: Literal["computer", "computer_use_preview"] = "computer",
    on_safety_check: Literal["refuse", "proceed"] = "refuse",
) -> Tool:
    """The ``computer`` tool, bound to ``session``.

    Args:
        session: The browser the actions are performed in.
        display_width: Viewport width, which is also the screenshot width.
        display_height: Viewport height.
        start_url: Opened before the first action; without it, open a page with
            the browser tools first.
        anthropic_version: Claude's computer tool version
            (see :data:`ANTHROPIC_TOOL_BETAS`).
        openai_tool: ``"computer"``, or ``"computer_use_preview"`` for models
            that only have the preview tool.
        on_safety_check: What to do with a call OpenAI flagged: ``"refuse"`` it,
            or ``"proceed"``, which acknowledges the checks. Gate the tool on
            ``computer-safety-check`` before choosing ``"proceed"``.
    """
    if anthropic_version not in ANTHROPIC_TOOL_BETAS:
        raise ValueError(
            f"unknown Anthropic computer tool version {anthropic_version!r}; "
            f"expected one of {sorted(ANTHROPIC_TOOL_BETAS)}"
        )
    if start_url is not None:
        session.check_url(start_url)
    screen = _Screen(session, display_width, display_height, start_url)

    async def computer(
        action: Any = None,
        actions: list[dict[str, Any]] | None = None,
        pending_safety_checks: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> str:
        if pending_safety_checks and on_safety_check != "proceed":
            flagged = "; ".join(
                f"{check.get('code', 'check')}: {check.get('message', '')}".strip()
                for check in pending_safety_checks
            )
            raise ComputerError(
                f"OpenAI flagged this action ({flagged}); it was not performed. "
                'Gate the computer tool on "computer-safety-check" and pass '
                'on_safety_check="proceed" to let an approver allow it.'
            )
        args = {"action": action, "actions": actions, **params}
        done = [await screen.perform(kind, step) for kind, step in _steps(args)]
        return await screen.report("; ".join(d for d in done if d))

    anthropic: dict[str, Any] = {
        "type": anthropic_version,
        "name": "computer",
        "display_width_px": display_width,
        "display_height_px": display_height,
        "_beta": ANTHROPIC_TOOL_BETAS[anthropic_version],
    }
    openai: dict[str, Any] = {"type": "computer"}
    if openai_tool == "computer_use_preview":
        openai = {
            "type": "computer_use_preview",
            "display_width": display_width,
            "display_height": display_height,
            "environment": "browser",
            "_request": {"truncation": "auto"},
        }
    return Tool(
        name="computer",
        description=_DESCRIPTION,
        parameters=_PARAMETERS,
        fn=computer,
        labels=READ_LABELS | WRITE_LABELS,
        native={"anthropic": anthropic, "openai_responses": openai},
    )


def _steps(args: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The actions in a call, as ``(kind, parameters)``, whichever provider sent it."""
    batched = args.get("actions")
    if isinstance(batched, list) and batched:
        return [_step(step) for step in batched]
    action = args.get("action")
    if isinstance(action, dict):
        return [_step(action)]
    if isinstance(action, str) and action:
        return [(action, {k: v for k, v in args.items() if k not in ("action", "actions")})]
    raise ComputerError("the computer tool needs an action")


def _step(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kind = action.get("type")
    if not isinstance(kind, str) or not kind:
        raise ComputerError(f"a computer action needs a type, got {action!r}")
    return kind, action


class _Screen:
    """Performs actions on the session's page and reports the result."""

    def __init__(
        self, session: BrowserSession, width: int, height: int, start_url: str | None
    ) -> None:
        self.session = session
        self.width = width
        self.height = height
        self.start_url = start_url
        self.cursor = (0, 0)
        self._ready = False

    async def page(self) -> Any:
        page = await self.session.page()
        if not self._ready:
            set_viewport = getattr(page, "set_viewport_size", None)
            if set_viewport is not None:
                await set_viewport({"width": self.width, "height": self.height})
            if self.start_url is not None:
                await page.goto(self.start_url)
                await self.session.ensure_allowed()
            self._ready = True
        return page

    def point(self, x: Any, y: Any) -> tuple[int, int]:
        if isinstance(x, bool) or isinstance(y, bool):
            raise ComputerError(f"({x}, {y}) is not a screen position")
        try:
            px, py = int(x), int(y)
        except (TypeError, ValueError) as exc:
            raise ComputerError(f"({x}, {y}) is not a screen position") from exc
        if not (0 <= px < self.width and 0 <= py < self.height):
            raise ComputerError(f"({px}, {py}) is off the {self.width}x{self.height} screen")
        return px, py

    def coordinate(self, value: Any) -> tuple[int, int]:
        if value is None:
            return self.cursor
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ComputerError(f"a coordinate is [x, y], not {value!r}")
        return self.point(value[0], value[1])

    async def move(self, page: Any, x: int, y: int) -> None:
        await page.mouse.move(x, y)
        self.cursor = (x, y)

    async def perform(self, kind: str, step: dict[str, Any]) -> str:  # noqa: C901, PLR0911, PLR0912
        page = await self.page()
        mouse, keyboard = page.mouse, page.keyboard

        if kind == "screenshot":
            return ""
        if kind == "cursor_position":
            return f"the pointer is at {self.cursor}"
        if kind == "wait":
            seconds = min(float(step.get("duration") or 1.0), _MAX_WAIT_SECONDS)
            await asyncio.sleep(seconds)
            return f"waited {seconds:g}s"

        if kind in ("mouse_move", "move"):
            x, y = self._target(step)
            await self.move(page, x, y)
            return f"moved to ({x}, {y})"

        if kind in (
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "click",
        ):
            x, y = self._target(step)
            button = _button(kind, step.get("button"))
            if button in ("back", "forward"):
                await (page.go_back() if button == "back" else page.go_forward())
                await self.session.ensure_allowed()
                return f"went {button}"
            count = {"double_click": 2, "triple_click": 3}.get(kind, 1)
            held = _modifiers(step)
            for key in held:
                await keyboard.down(key)
            try:
                await self.move(page, x, y)
                await mouse.click(x, y, button=button, click_count=count)
            finally:
                for key in reversed(held):
                    await keyboard.up(key)
            await self.session.ensure_allowed()
            return f"{kind.replace('_', ' ')} at ({x}, {y})"

        if kind == "left_click_drag":
            sx, sy = self.coordinate(step.get("start_coordinate"))
            ex, ey = self.coordinate(step.get("coordinate"))
            return await self._drag(page, [(sx, sy), (ex, ey)])
        if kind == "drag":
            path = [self.point(p.get("x"), p.get("y")) for p in step.get("path") or []]
            if len(path) < 2:
                raise ComputerError("a drag needs a path of at least two points")
            return await self._drag(page, path)
        if kind == "left_mouse_down":
            await mouse.down()
            return "pressed the left button"
        if kind == "left_mouse_up":
            await mouse.up()
            await self.session.ensure_allowed()
            return "released the left button"

        if kind == "type":
            text = str(step.get("text") or "")
            await keyboard.type(text)
            return f"typed {len(text)} characters"
        if kind in ("key", "keypress"):
            combo = _combo(step.get("keys") if kind == "keypress" else step.get("text"))
            await keyboard.press(combo)
            await self.session.ensure_allowed()
            return f"pressed {combo}"
        if kind == "hold_key":
            combo = _combo(step.get("text"))
            seconds = min(float(step.get("duration") or 0.5), _MAX_WAIT_SECONDS)
            keys = combo.split("+")
            for key in keys:
                await keyboard.down(key)
            await asyncio.sleep(seconds)
            for key in reversed(keys):
                await keyboard.up(key)
            return f"held {combo} for {seconds:g}s"

        if kind == "scroll":
            x, y = self._target(step)
            await self.move(page, x, y)
            if "scroll_x" in step or "scroll_y" in step:
                dx, dy = int(step.get("scroll_x") or 0), int(step.get("scroll_y") or 0)
            else:
                amount = int(step.get("scroll_amount") or 3) * _WHEEL_PIXELS
                dx, dy = {
                    "up": (0, -amount),
                    "down": (0, amount),
                    "left": (-amount, 0),
                    "right": (amount, 0),
                }.get(str(step.get("scroll_direction") or "down"), (0, amount))
            await mouse.wheel(dx, dy)
            return f"scrolled ({dx}, {dy}) at ({x}, {y})"

        raise ComputerError(f"the computer tool does not support the {kind!r} action")

    def _target(self, step: dict[str, Any]) -> tuple[int, int]:
        if "x" in step or "y" in step:
            return self.point(step.get("x"), step.get("y"))
        return self.coordinate(step.get("coordinate"))

    async def _drag(self, page: Any, path: list[tuple[int, int]]) -> str:
        await self.move(page, *path[0])
        await page.mouse.down()
        for x, y in path[1:]:
            await self.move(page, x, y)
        await page.mouse.up()
        await self.session.ensure_allowed()
        return f"dragged from {path[0]} to {path[-1]}"

    async def report(self, done: str) -> str:
        page = await self.page()
        png = await page.screenshot(type="png")
        summary = f"{done}; " if done else ""
        return f"{summary}the page is at {page.url}" + encode_image(png)


def _button(kind: str, button: Any) -> str:
    if kind == "right_click":
        return "right"
    if kind == "middle_click":
        return "middle"
    if button in (None, "left"):
        return "left"
    if button == "wheel":
        return "middle"
    if button in ("right", "back", "forward"):
        return str(button)
    raise ComputerError(f"unknown mouse button {button!r}")


def _key(name: str) -> str:
    stripped = name.strip()
    mapped = _KEYS.get(stripped.lower())
    if mapped is not None:
        return mapped
    if len(stripped) > 1 and stripped[0] in "fF" and stripped[1:].isdigit():
        return stripped.upper()
    return stripped


def _combo(keys: Any) -> str:
    """Keys as a Playwright combination: ``"ctrl+a"`` or ``["CTRL", "A"]`` -> ``"Control+a"``."""
    names = keys if isinstance(keys, list) else str(keys or "").split("+")
    mapped = [_key(str(k)) for k in names if str(k).strip()]
    if not mapped:
        raise ComputerError("a key action needs keys")
    if any(k in _MODIFIERS for k in mapped):
        mapped = [k.lower() if len(k) == 1 else k for k in mapped]
    return "+".join(mapped)


def _modifiers(step: dict[str, Any]) -> list[str]:
    """Keys held during a click: Anthropic's ``text`` or OpenAI's ``keys``."""
    raw = step.get("keys")
    if raw is None and step.get("text"):
        raw = str(step["text"]).split("+")
    return [_key(str(k)) for k in raw or [] if str(k).strip()]


__all__ = [
    "ANTHROPIC_TOOL_BETAS",
    "READ_ACTIONS",
    "READ_LABELS",
    "SAFETY_CHECK_LABEL",
    "WRITE_LABELS",
    "ComputerError",
    "computer_action",
    "computer_tool",
]

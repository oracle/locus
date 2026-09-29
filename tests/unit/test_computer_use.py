# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Computer use: the tool against a fake page, and what each adapter sends.

The live behaviour against real Chromium is in
``tests/integration/test_computer_use_live.py``.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tulip.control import ControlPolicy, gate_tool
from tulip.core.media import (
    EARLIER_IMAGE_OMITTED,
    IMAGE_OMITTED,
    IMAGE_TOKEN_ESTIMATE,
    encode_image,
    estimate_tokens,
    has_images,
    images,
    recent_image_positions,
    split_content,
    strip_images,
    text_length,
)
from tulip.core.messages import Message, Role, ToolCall, ToolResult
from tulip.models.native.anthropic import AnthropicModel
from tulip.models.native.openai import (
    COMPUTER_ITEM_ID_ARG,
    RESPONSES_ITEMS_METADATA_KEY,
    OpenAIModel,
)
from tulip.tools.browser import BrowserError, BrowserSession
from tulip.tools.computer import (
    READ_LABELS,
    SAFETY_CHECK_LABEL,
    WRITE_LABELS,
    ComputerError,
    _combo,
    computer_action,
    computer_tool,
)


PNG = b"\x89PNG\r\n\x1a\nfake-screenshot"


# ---------------------------------------------------------------------------
# tulip.core.media
# ---------------------------------------------------------------------------


class TestMedia:
    def test_round_trip(self) -> None:
        content = "clicked" + encode_image(PNG) + "after"
        parts = split_content(content)
        assert parts[0] == "clicked"
        assert parts[1].media_type == "image/png"  # type: ignore[union-attr]
        assert base64.b64decode(parts[1].data) == PNG  # type: ignore[union-attr]
        assert parts[2] == "after"
        assert images(content)[0].data_url.startswith("data:image/png;base64,")

    def test_plain_text_is_untouched(self) -> None:
        assert not has_images("plain")
        assert not has_images(None)
        assert images("plain") == []
        assert strip_images("plain") == "plain"
        assert text_length("plain") == 5
        assert text_length(None) == 0

    def test_a_forged_marker_with_bad_base64_stays_text(self) -> None:
        forged = "x\n[tulip-image media_type=image/png]\n@@notbase64@@\n[/tulip-image]\n"
        assert has_images(forged)
        assert all(isinstance(part, str) for part in split_content(forged))
        assert images(forged) == []

    def test_measures_text_not_images(self) -> None:
        content = "ab" + encode_image(PNG * 100)
        assert text_length(content) == 2
        assert estimate_tokens(content) == IMAGE_TOKEN_ESTIMATE
        assert strip_images(content) == f"ab\n{IMAGE_OMITTED}"

    def test_recent_image_positions(self) -> None:
        shot = encode_image(PNG)
        contents = [shot, None, "text", shot, shot, shot]
        assert recent_image_positions(contents) == {3, 4, 5}
        assert recent_image_positions(contents, keep=1) == {5}
        assert recent_image_positions(contents, keep=0) == set()


# ---------------------------------------------------------------------------
# The tool against a fake page
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, page: _Page, name: str) -> None:
        self.page = page
        self.name = name

    def __getattr__(self, method: str) -> Any:
        async def record(*args: Any, **kwargs: Any) -> None:
            self.page.calls.append((f"{self.name}.{method}", args, kwargs))
            if self.name == "mouse" and method == "click":
                target = self.page.links.get((args[0], args[1]))
                if target:
                    self.page.url = target

        return record


class _Page:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.links: dict[tuple[int, int], str] = {}
        self.mouse = _Recorder(self, "mouse")
        self.keyboard = _Recorder(self, "keyboard")

    async def goto(self, url: str) -> None:
        self.calls.append(("goto", (url,), {}))
        self.url = url

    async def set_viewport_size(self, size: dict[str, int]) -> None:
        self.calls.append(("viewport", (size,), {}))

    async def screenshot(self, **kwargs: Any) -> bytes:
        return PNG

    async def go_back(self) -> None:
        self.calls.append(("back", (), {}))

    async def go_forward(self) -> None:
        self.calls.append(("forward", (), {}))

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]


def _computer(**options: Any) -> tuple[Any, _Page]:
    page = _Page()

    async def factory() -> _Page:
        return page

    session = BrowserSession(allowed_domains=["shop.example.com"], page_factory=factory)
    options.setdefault("start_url", "https://shop.example.com/orders")
    return computer_tool(session, **options), page


class TestDefinition:
    def test_native_definitions(self) -> None:
        tool, _ = _computer()
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "computer"
        assert schema["native"]["anthropic"] == {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1280,
            "display_height_px": 800,
            "_beta": "computer-use-2025-11-24",
        }
        assert schema["native"]["openai_responses"] == {"type": "computer"}
        assert tool.labels == READ_LABELS | WRITE_LABELS

    def test_preview_tool_and_older_anthropic_version(self) -> None:
        tool, _ = _computer(
            openai_tool="computer_use_preview",
            anthropic_version="computer_20250124",
            display_width=1024,
            display_height=768,
        )
        assert tool.native["openai_responses"] == {
            "type": "computer_use_preview",
            "display_width": 1024,
            "display_height": 768,
            "environment": "browser",
            "_request": {"truncation": "auto"},
        }
        assert tool.native["anthropic"]["_beta"] == "computer-use-2025-01-24"

    def test_rejects_unknown_version_and_disallowed_start(self) -> None:
        with pytest.raises(ValueError, match="unknown Anthropic"):
            _computer(anthropic_version="computer_1999")
        with pytest.raises(BrowserError):
            _computer(start_url="https://evil.example.net/")

    def test_plain_tools_have_no_native_key(self) -> None:
        from tulip.tools.decorator import tool as tool_decorator

        @tool_decorator
        def ping() -> str:
            """Ping."""
            return "pong"

        assert "native" not in ping.to_openai_schema()


class TestAnthropicActions:
    @pytest.mark.asyncio
    async def test_screenshot_opens_start_url_once(self) -> None:
        tool, page = _computer()
        result = await tool.execute(action="screenshot")
        await tool.execute(action="screenshot")
        assert page.names().count("goto") == 1
        assert page.names().count("viewport") == 1
        assert "the page is at https://shop.example.com/orders" in result
        assert base64.b64decode(images(result)[0].data) == PNG

    @pytest.mark.asyncio
    async def test_click_with_modifiers(self) -> None:
        tool, page = _computer()
        result = await tool.execute(action="left_click", coordinate=[10, 20], text="ctrl+shift")
        assert result.startswith("left click at (10, 20)")
        downs = [args[0] for name, args, _ in page.calls if name == "keyboard.down"]
        ups = [args[0] for name, args, _ in page.calls if name == "keyboard.up"]
        assert downs == ["Control", "Shift"]
        assert ups == ["Shift", "Control"]
        click = next(c for c in page.calls if c[0] == "mouse.click")
        assert click[1] == (10, 20)
        assert click[2] == {"button": "left", "click_count": 1}

    @pytest.mark.asyncio
    async def test_other_clicks_use_the_cursor(self) -> None:
        tool, page = _computer()
        await tool.execute(action="mouse_move", coordinate=[5, 6])
        await tool.execute(action="right_click")
        await tool.execute(action="middle_click")
        await tool.execute(action="triple_click", coordinate=[7, 8])
        clicks = [(args, kwargs) for name, args, kwargs in page.calls if name == "mouse.click"]
        assert clicks == [
            ((5, 6), {"button": "right", "click_count": 1}),
            ((5, 6), {"button": "middle", "click_count": 1}),
            ((7, 8), {"button": "left", "click_count": 3}),
        ]
        assert "(7, 8)" in await tool.execute(action="cursor_position")

    @pytest.mark.asyncio
    async def test_drag_type_key_hold_scroll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tulip.tools.computer.asyncio.sleep", AsyncMock())
        tool, page = _computer()
        await tool.execute(action="left_click_drag", start_coordinate=[1, 2], coordinate=[3, 4])
        await tool.execute(action="type", text="hello")
        await tool.execute(action="key", text="ctrl+a")
        await tool.execute(action="hold_key", text="shift", duration=1)
        await tool.execute(
            action="scroll", coordinate=[50, 60], scroll_direction="up", scroll_amount=2
        )
        await tool.execute(action="scroll", coordinate=[50, 60], scroll_direction="left")
        await tool.execute(action="left_mouse_down")
        await tool.execute(action="left_mouse_up")
        waited = await tool.execute(action="wait", duration=99)
        calls = [(name, args) for name, args, _ in page.calls if name != "mouse.move"]
        assert ("mouse.down", ()) in calls
        assert ("keyboard.type", ("hello",)) in calls
        assert ("keyboard.press", ("Control+a",)) in calls
        assert ("keyboard.down", ("Shift",)) in calls
        assert ("mouse.wheel", (0, -200)) in calls
        assert ("mouse.wheel", (-300, 0)) in calls
        assert waited.startswith("waited 10s")

    @pytest.mark.asyncio
    async def test_refusals(self) -> None:
        tool, _ = _computer()
        with pytest.raises(ComputerError, match="off the 1280x800 screen"):
            await tool.execute(action="left_click", coordinate=[5000, 1])
        with pytest.raises(ComputerError, match="not a screen position"):
            await tool.execute(action="left_click", coordinate=["a", 1])
        with pytest.raises(ComputerError, match="not a screen position"):
            await tool.execute(action="left_click", coordinate=[True, 1])
        with pytest.raises(ComputerError, match="coordinate is"):
            await tool.execute(action="left_click", coordinate=[1])
        with pytest.raises(ComputerError, match="needs an action"):
            await tool.execute()
        with pytest.raises(ComputerError, match="does not support"):
            await tool.execute(action="zoom")
        with pytest.raises(ComputerError, match="needs keys"):
            await tool.execute(action="key", text="")

    @pytest.mark.asyncio
    async def test_a_click_that_leaves_the_allowed_domains(self) -> None:
        tool, page = _computer()
        page.links[(9, 9)] = "https://evil.example.net/"
        with pytest.raises(BrowserError, match="not allowed"):
            await tool.execute(action="left_click", coordinate=[9, 9])
        assert page.url == "about:blank"


class TestOpenAIActions:
    @pytest.mark.asyncio
    async def test_single_actions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tulip.tools.computer.asyncio.sleep", AsyncMock())
        tool, page = _computer()
        await tool.execute(action={"type": "click", "x": 1, "y": 2, "button": "wheel"})
        await tool.execute(action={"type": "double_click", "x": 3, "y": 4})
        await tool.execute(
            action={"type": "drag", "path": [{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}]}
        )
        await tool.execute(action={"type": "keypress", "keys": ["CTRL", "L"]})
        await tool.execute(
            action={"type": "scroll", "x": 10, "y": 10, "scroll_x": 0, "scroll_y": 400}
        )
        await tool.execute(action={"type": "move", "x": 8, "y": 8})
        await tool.execute(action={"type": "type", "text": "hi"})
        await tool.execute(action={"type": "wait"})
        await tool.execute(action={"type": "screenshot"})
        await tool.execute(action={"type": "click", "x": 1, "y": 1, "button": "back"})
        await tool.execute(action={"type": "click", "x": 1, "y": 1, "button": "forward"})
        names = page.names()
        assert names.count("mouse.move") >= 5
        assert ("mouse.click", (1, 2), {"button": "middle", "click_count": 1}) in page.calls
        assert ("mouse.click", (3, 4), {"button": "left", "click_count": 2}) in page.calls
        assert ("keyboard.press", ("Control+l",), {}) in page.calls
        assert ("mouse.wheel", (0, 400), {}) in page.calls
        assert "back" in names
        assert "forward" in names

    @pytest.mark.asyncio
    async def test_batched_actions(self) -> None:
        tool, page = _computer()
        result = await tool.execute(
            actions=[{"type": "click", "x": 1, "y": 1}, {"type": "type", "text": "abc"}],
            openai_item_id="cu_1",
        )
        assert result.startswith("click at (1, 1); typed 3 characters;")
        assert has_images(result)

    @pytest.mark.asyncio
    async def test_bad_actions(self) -> None:
        tool, _ = _computer()
        with pytest.raises(ComputerError, match="needs a type"):
            await tool.execute(action={"x": 1})
        with pytest.raises(ComputerError, match="two points"):
            await tool.execute(action={"type": "drag", "path": [{"x": 1, "y": 1}]})
        with pytest.raises(ComputerError, match="unknown mouse button"):
            await tool.execute(action={"type": "click", "x": 1, "y": 1, "button": "thumb"})

    @pytest.mark.asyncio
    async def test_safety_checks_are_refused_unless_told_to_proceed(self) -> None:
        checks = [{"id": "sc_1", "code": "malicious_instructions", "message": "odd page"}]
        tool, page = _computer()
        with pytest.raises(ComputerError, match="malicious_instructions: odd page"):
            await tool.execute(
                action={"type": "click", "x": 1, "y": 1}, pending_safety_checks=checks
            )
        assert "mouse.click" not in page.names()

        proceeding, page = _computer(on_safety_check="proceed")
        await proceeding.execute(
            action={"type": "click", "x": 1, "y": 1}, pending_safety_checks=checks
        )
        assert "mouse.click" in page.names()

    def test_key_names(self) -> None:
        assert _combo("Return") == "Enter"
        assert _combo("ctrl+A") == "Control+a"
        assert _combo(["CMD", "SHIFT", "T"]) == "Meta+Shift+t"
        assert _combo("f5") == "F5"
        assert _combo(["ARROWDOWN"]) == "ArrowDown"
        assert _combo("x") == "x"


class TestPolicy:
    def test_action_tags(self) -> None:
        read = computer_action("computer", {"action": "screenshot"})
        assert read.tags == READ_LABELS
        assert read.asset == "screenshot"
        write = computer_action("computer", {"actions": [{"type": "move"}, {"type": "click"}]})
        assert write.tags == WRITE_LABELS
        assert write.asset == "move,click"
        flagged = computer_action(
            "computer", {"action": {"type": "screenshot"}, "pending_safety_checks": [{"id": "x"}]}
        )
        assert SAFETY_CHECK_LABEL in flagged.tags

    @pytest.mark.asyncio
    async def test_a_gated_write_does_not_touch_the_page(self) -> None:
        tool, page = _computer()
        gated = gate_tool(
            tool,
            policy=ControlPolicy(require_human_for=frozenset({"computer-write"})),
            action=computer_action,
        )
        refused = await gated.execute(action="left_click", coordinate=[1, 1])
        assert "mouse.click" not in page.names()
        assert "held_for_approval" in str(refused)
        assert not has_images(str(refused))


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


def _native_schemas() -> list[dict[str, Any]]:
    tool, _ = _computer()
    return [tool.to_openai_schema()]


def _tool_message(call_id: str, content: str) -> Message:
    return Message.tool(ToolResult(tool_call_id=call_id, name="computer", content=content))


class TestAnthropicAdapter:
    def test_native_tool_and_beta(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tools = m._convert_tools(_native_schemas())
        assert tools == [
            {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": 1280,
                "display_height_px": 800,
            }
        ]
        assert m._native_betas(_native_schemas()) == {"anthropic-beta": "computer-use-2025-11-24"}
        assert m._native_betas([{"type": "function", "function": {"name": "f"}}]) == {}
        assert m._native_betas(None) == {}

    def test_screenshots_become_image_blocks_and_old_ones_placeholders(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        shot = "done" + encode_image(PNG)
        messages = [Message.user("go")] + [_tool_message(f"t{i}", shot) for i in range(4)]
        _, converted = m._convert_messages(messages)
        oldest = converted[1]["content"][0]["content"]
        assert oldest == f"done\n{EARLIER_IMAGE_OMITTED}"
        latest = converted[-1]["content"][0]["content"]
        assert latest[0] == {"type": "text", "text": "done"}
        assert latest[1]["type"] == "image"
        assert latest[1]["source"]["media_type"] == "image/png"
        assert base64.b64decode(latest[1]["source"]["data"]) == PNG
        assert isinstance(converted[2]["content"][0]["content"], list)

    @pytest.mark.asyncio
    async def test_complete_sends_the_beta_header(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="toolu_1",
                        name="computer",
                        input={"action": "left_click", "coordinate": [1, 2]},
                    )
                ],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="tool_use",
            )
        )
        m._client = client  # type: ignore[assignment]
        response = await m.complete([Message.user("click")], tools=_native_schemas())
        params = client.messages.create.call_args.kwargs
        assert params["extra_headers"] == {"anthropic-beta": "computer-use-2025-11-24"}
        assert params["tools"][0]["type"] == "computer_20251124"
        assert response.message.tool_calls[0].arguments == {
            "action": "left_click",
            "coordinate": [1, 2],
        }

    @pytest.mark.asyncio
    async def test_stream_sends_the_beta_header(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        seen: dict[str, Any] = {}

        class _TextStream:
            async def __aenter__(self) -> _TextStream:
                self.text_stream = self._texts()
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            async def _texts(self) -> Any:
                yield "hi"

            async def get_final_message(self) -> Any:
                return SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

        def stream(**params: Any) -> _TextStream:
            seen.update(params)
            return _TextStream()

        client = MagicMock()
        client.messages.stream = stream
        m._client = client  # type: ignore[assignment]
        chunks = [chunk async for chunk in m.stream([Message.user("hi")], tools=_native_schemas())]
        assert seen["extra_headers"] == {"anthropic-beta": "computer-use-2025-11-24"}
        assert chunks[0].content == "hi"


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class _Item:
    def __init__(self, dump: dict[str, Any]) -> None:
        self._dump = dump
        for key, value in dump.items():
            setattr(self, key, value)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._dump


_CALL = {
    "type": "computer_call",
    "id": "cu_1",
    "call_id": "call_c1",
    "status": "completed",
    "action": {"type": "click", "x": 1, "y": 2, "button": "left"},
    "pending_safety_checks": [{"id": "sc_1", "code": "irrelevant_domain", "message": "hm"}],
}


class TestOpenAIChatAdapter:
    def test_native_key_and_images_do_not_reach_chat_completions(self) -> None:
        m = OpenAIModel(model="gpt-4o", api_key="sk-x")  # noqa: S106
        tools = m._convert_tools(_native_schemas())
        assert tools is not None
        assert "native" not in tools[0]
        assert tools[0]["function"]["name"] == "computer"
        converted = m._convert_messages(
            [Message.user("go"), _tool_message("t1", "done" + encode_image(PNG))]
        )
        assert converted[-1]["content"] == f"done\n{IMAGE_OMITTED}"


class TestOpenAIResponsesAdapter:
    def _model(self) -> OpenAIModel:
        return OpenAIModel(model="gpt-5.6-sol", api_key="sk-x")  # noqa: S106

    def test_native_tool_and_request_option(self) -> None:
        m = self._model()
        plain = {"type": "function", "function": {"name": "f", "parameters": {}}}
        assert m._convert_tools_responses([*_native_schemas(), plain]) == [
            {"type": "computer"},
            {"type": "function", "name": "f", "parameters": {}},
        ]
        preview, _ = _computer(openai_tool="computer_use_preview")
        request = m._build_responses_request(
            [Message.user("go")], [preview.to_openai_schema()], {}, stream=False
        )
        assert request["truncation"] == "auto"
        assert request["tools"][0]["type"] == "computer_use_preview"
        explicit = m._build_responses_request(
            [Message.user("go")],
            [preview.to_openai_schema()],
            {"truncation": "disabled"},
            stream=False,
        )
        assert explicit["truncation"] == "disabled"

    def test_computer_call_is_a_call_to_the_computer_tool(self) -> None:
        m = self._model()
        response = SimpleNamespace(
            output=[_Item(_CALL)], usage=None, status="completed", incomplete_details=None
        )
        parsed = m._parse_responses_result(response)
        call = parsed.message.tool_calls[0]
        assert call.id == "call_c1"
        assert call.name == "computer"
        assert call.arguments == {
            "action": _CALL["action"],
            "pending_safety_checks": _CALL["pending_safety_checks"],
            COMPUTER_ITEM_ID_ARG: "cu_1",
        }
        assert parsed.message.metadata[RESPONSES_ITEMS_METADATA_KEY] == [_CALL]

    def test_screenshot_result_acknowledges_checks(self) -> None:
        m = self._model()
        assistant = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="call_c1", name="computer", arguments={})],
            metadata={RESPONSES_ITEMS_METADATA_KEY: [_CALL]},
        )
        items = m._convert_messages_responses(
            [Message.user("go"), assistant, _tool_message("call_c1", "clicked" + encode_image(PNG))]
        )
        assert items[1] == _CALL
        output = items[2]
        assert output["type"] == "computer_call_output"
        assert output["call_id"] == "call_c1"
        assert output["output"]["image_url"].startswith("data:image/png;base64,")
        assert output["acknowledged_safety_checks"] == _CALL["pending_safety_checks"]
        assert items[3] == {"role": "user", "content": "[computer result] clicked"}

    def test_a_call_that_did_not_run_acknowledges_nothing(self) -> None:
        m = self._model()
        assistant = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="call_c1", name="computer", arguments={})],
            metadata={RESPONSES_ITEMS_METADATA_KEY: [_CALL]},
        )
        items = m._convert_messages_responses(
            [Message.user("go"), assistant, _tool_message("call_c1", "Refused: needs approval")]
        )
        output = items[2]
        assert "acknowledged_safety_checks" not in output
        assert output["output"]["image_url"].startswith("data:image/png;base64,iVBOR")
        assert items[3]["content"] == "[computer result] Refused: needs approval"

    def test_rebuilt_turn_replays_the_computer_call(self) -> None:
        m = self._model()
        arguments = {"actions": [{"type": "screenshot"}], COMPUTER_ITEM_ID_ARG: "cu_9"}
        assistant = Message.assistant(
            content=None, tool_calls=[ToolCall(id="call_9", name="computer", arguments=arguments)]
        )
        messages = [Message.user("go"), assistant]
        messages += [_tool_message("call_9", encode_image(PNG))]
        items = m._convert_messages_responses(messages)
        assert items[1] == {
            "type": "computer_call",
            "id": "cu_9",
            "call_id": "call_9",
            "status": "completed",
            "pending_safety_checks": [],
            "actions": [{"type": "screenshot"}],
        }
        assert items[2]["type"] == "computer_call_output"
        assert len(items) == 3

    def test_old_screenshots_and_image_function_outputs(self) -> None:
        m = self._model()
        shot = encode_image(PNG)
        calls = [
            ToolCall(id=f"call_{i}", name="computer", arguments={COMPUTER_ITEM_ID_ARG: f"cu_{i}"})
            for i in range(4)
        ]
        messages = [Message.user("go"), Message.assistant(content=None, tool_calls=calls)]
        messages += [_tool_message(f"call_{i}", shot) for i in range(4)]
        items = m._convert_messages_responses(messages)
        outputs = [i for i in items if i.get("type") == "computer_call_output"]
        assert outputs[0]["output"]["image_url"].startswith("data:image/png;base64,iVBOR")
        assert outputs[-1]["output"]["image_url"] == images(shot)[0].data_url

        chart = Message.tool(
            ToolResult(tool_call_id="call_f", name="chart", content="chart" + shot)
        )
        fn_call = Message.assistant(
            content=None, tool_calls=[ToolCall(id="call_f", name="chart", arguments={})]
        )
        items = m._convert_messages_responses([Message.user("go"), fn_call, chart])
        assert items[-1]["output"] == [
            {"type": "input_text", "text": "chart"},
            {"type": "input_image", "image_url": images(shot)[0].data_url},
        ]
        stale = [Message.user("go"), fn_call, chart] + [
            _tool_message("call_x", shot) for _ in range(3)
        ]
        items = m._convert_messages_responses(stale)
        assert items[2]["output"] == f"chart\n{EARLIER_IMAGE_OMITTED}"
        plain = [
            Message.user("go"),
            fn_call,
            Message.tool(ToolResult(tool_call_id="call_f", name="chart", content="ok")),
        ]
        assert m._convert_messages_responses(plain)[-1]["output"] == "ok"

    @pytest.mark.asyncio
    async def test_streamed_computer_call(self) -> None:
        m = self._model()

        async def events() -> Any:
            yield SimpleNamespace(type="response.output_item.done", item=_Item(_CALL))
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=None, status="completed", incomplete_details=None, output=[]
                ),
            )

        client = MagicMock()
        client.responses.create = AsyncMock(return_value=events())
        m._client = client  # type: ignore[assignment]
        chunks = [chunk async for chunk in m.stream([Message.user("go")], tools=_native_schemas())]
        calls = [call for chunk in chunks for call in (chunk.tool_calls or [])]
        assert calls[0].name == "computer"
        assert calls[0].arguments[COMPUTER_ITEM_ID_ARG] == "cu_1"
        sent = client.responses.create.call_args.kwargs
        assert sent["tools"] == [{"type": "computer"}]
        assert json.dumps(sent)  # the request is plain JSON


# ---------------------------------------------------------------------------
# The loop and the estimates measure text, not screenshots
# ---------------------------------------------------------------------------


class _ScriptedModel:
    name = "stub"

    def __init__(self, tool_name: str) -> None:
        from tulip.models import ModelResponse

        self._scripted = [
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="t1", name=tool_name, arguments={})]
                )
            ),
            ModelResponse(message=Message.assistant("done")),
        ]

    async def complete(self, messages: list[Message], tools: Any = None, **kwargs: Any) -> Any:
        return self._scripted.pop(0)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


_BIG_SHOT = encode_image(PNG * 400)  # far over the 2,000-character cap


def _tool_content(tool_fn: Any) -> str:
    from tulip.agent.agent import Agent
    from tulip.agent.config import AgentConfig

    agent = Agent(
        config=AgentConfig(
            model=_ScriptedModel(tool_fn.name),
            tools=[tool_fn],
            max_iterations=3,
            max_tool_result_length=2_000,
        )
    )
    result = agent.run_sync("go")
    return next(m.content or "" for m in result.state.messages if m.role == Role.TOOL)


class TestLoopAndEstimates:
    def test_a_screenshot_is_not_truncated(self) -> None:
        from tulip.tools.decorator import tool as tool_decorator

        @tool_decorator
        def snap() -> str:
            """Take a screenshot."""
            return "the page is at https://shop.example.com/" + _BIG_SHOT

        content = _tool_content(snap)
        assert "[OUTPUT TRUNCATED" not in content
        assert base64.b64decode(images(content)[0].data) == PNG * 400

    def test_long_text_is_truncated_without_corrupting_an_image(self) -> None:
        from tulip.tools.decorator import tool as tool_decorator

        @tool_decorator
        def dump() -> str:
            """Dump a page."""
            return "x" * 5_000 + _BIG_SHOT

        content = _tool_content(dump)
        assert "[OUTPUT TRUNCATED" in content
        assert images(content) == []
        assert len(content) < 2_100

    def test_token_estimates_count_images_as_images(self) -> None:
        from tulip.core.state import AgentState
        from tulip.memory.compactor import _char_count_tokens

        message = _tool_message("t1", "abcd" + _BIG_SHOT)
        assert _char_count_tokens(message) == 1 + IMAGE_TOKEN_ESTIMATE
        state = AgentState(messages=[message])
        assert state.total_tokens == 1 + IMAGE_TOKEN_ESTIMATE

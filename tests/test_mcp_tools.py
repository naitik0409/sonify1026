"""Tests for the MCP tool helpers and server wiring.

The generate test needs a network connection (audio is served by edge-tts),
matching the rest of the suite. The server test is skipped if `mcp` is not
installed.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import snfy.mcp_server as mcp_server


def test_list_emotions():
    emotions = mcp_server.list_emotions()
    assert len(emotions) == 17
    assert "neutral" in emotions
    assert "warm" in emotions


def test_resolve_language():
    from snfy import Language

    assert mcp_server._resolve_language("auto") is None
    assert mcp_server._resolve_language("") is None
    assert mcp_server._resolve_language("hindi") is Language.HI_IN
    assert mcp_server._resolve_language("hi-in") is Language.HI_IN
    assert mcp_server._resolve_language("us") is Language.EN_US
    assert mcp_server._resolve_language("english") is Language.EN_IN


def test_synthesize_to_file_writes_audio(tmp_path):
    out = tmp_path / "mcp_out.wav"
    result = mcp_server.synthesize_to_file(
        "Hello from the MCP test suite.",
        emotion="neutral",
        output_path=str(out),
    )
    assert os.path.exists(result["path"])
    assert result["path"] == str(out)
    assert result["sample_rate"] == 48000
    assert result["duration"] > 0
    assert result["emotion"] == "neutral"


def test_synthesize_rejects_empty_text():
    with pytest.raises(ValueError):
        mcp_server.synthesize_to_file("   ")


def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    server = mcp_server.build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"synthesize", "emotions"} <= names

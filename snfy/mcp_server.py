"""Model Context Protocol (MCP) server for Sonify 1026.

Exposes the voice engine as MCP tools so any MCP client (Claude Desktop,
Cursor, Windsurf, custom agents, ...) can give itself an expressive Indian
female voice.

Install the extra and run:

    pip install "sonify1026[mcp]"
    snfy-mcp                       # or: python -m snfy.mcp_server

The core helpers (`synthesize_to_file`, `list_emotions`) do not import the
`mcp` package, so they are usable (and testable) without it.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

from .engine import Emotion, Language, SonifyEngine

_LANGUAGES = {
    "auto": None,
    "": None,
    "english": Language.EN_IN,
    "en": Language.EN_IN,
    "en-in": Language.EN_IN,
    "en_in": Language.EN_IN,
    "indian english": Language.EN_IN,
    "hindi": Language.HI_IN,
    "hi": Language.HI_IN,
    "hi-in": Language.HI_IN,
    "hi_in": Language.HI_IN,
    "us": Language.EN_US,
    "en-us": Language.EN_US,
    "en_us": Language.EN_US,
    "american": Language.EN_US,
}

_engine: Optional[SonifyEngine] = None


def _get_engine() -> SonifyEngine:
    global _engine
    if _engine is None:
        _engine = SonifyEngine()
    return _engine


def _resolve_language(language: Optional[str]):
    if language is None:
        return None
    return _LANGUAGES.get(str(language).strip().lower())


def list_emotions() -> list[str]:
    """Return the names of all available emotion presets."""
    return [e.value for e in Emotion]


def synthesize_to_file(
    text: str,
    emotion: str = "neutral",
    language: str = "auto",
    speed: float = 1.0,
    pitch: float = 0.0,
    output_path: Optional[str] = None,
) -> dict:
    """Render `text` to a WAV file and return a small result dict.

    `output_path` defaults to `$SNFY_OUTPUT_DIR` (or a temp dir) with a
    timestamped filename, so repeated calls never overwrite each other.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    engine = _get_engine()
    engine.set_speed(float(speed))
    engine.set_pitch(float(pitch))

    if not output_path:
        out_dir = os.environ.get("SNFY_OUTPUT_DIR") or os.path.join(
            tempfile.gettempdir(), "snfy"
        )
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"snfy_{int(time.time() * 1000)}.wav")

    result = engine.synthesize(
        text.strip(),
        emotion=emotion,
        language=_resolve_language(language),
        output_path=output_path,
    )
    return {
        "path": os.path.abspath(output_path),
        "duration": round(float(result.duration), 2),
        "sample_rate": int(result.sample_rate),
        "emotion": getattr(result.emotion, "value", result.emotion),
        "language": getattr(result.language, "value", result.language),
    }


def build_server():
    """Create and return an MCP server exposing the voice tools.

    Works with both the MCP SDK 2.x (`mcp.server.mcpserver.MCPServer`) and
    1.x (`mcp.server.fastmcp.FastMCP`).
    """
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except Exception:
        from mcp.server.fastmcp import FastMCP as _Server

    mcp = _Server("sonify1026")

    @mcp.tool()
    def synthesize(
        text: str,
        emotion: str = "neutral",
        language: str = "auto",
        speed: float = 1.0,
        pitch: float = 0.0,
    ) -> dict:
        """Speak `text` aloud and save it as a WAV file.

        Args:
            text: What to say (English, Hindi or Hinglish).
            emotion: One of the Sonify emotion presets (see `list_emotions`).
            language: "auto" (default), "english"/"en-in", "hindi"/"hi-in" or "us".
            speed: 0.5 to 2.0 (1.0 = normal).
            pitch: -20 to +25 Hz (0 = normal).

        Returns:
            A dict with the saved `path`, plus duration, sample_rate, emotion
            and language. Play or attach the file at `path`.
        """
        return synthesize_to_file(text, emotion, language, speed, pitch)

    @mcp.tool()
    def emotions() -> list[str]:
        """List the available emotion presets for the `synthesize` tool."""
        return list_emotions()

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()

"""Use Sonify 1026 as a LlamaIndex tool.

    pip install llama-index-core "sonify1026[mcp]"

`speak_tool` is a FunctionTool you can hand to any LlamaIndex agent or query
engine. It returns the absolute path of the generated WAV file.
"""

from __future__ import annotations

from llama_index.core.tools import FunctionTool

from snfy.mcp_server import synthesize_to_file


def speak(text: str, emotion: str = "neutral", language: str = "auto") -> str:
    """Speak text aloud with an expressive Indian female voice; returns the WAV path.

    Args:
        text: The words to speak (English, Hindi or Hinglish).
        emotion: A Sonify preset, e.g. "warm", "happy", "calm", "whisper".
        language: "auto" (default), "english", "hindi" or "us".
    """
    return synthesize_to_file(text, emotion=emotion, language=language)["path"]


speak_tool = FunctionTool.from_defaults(fn=speak)


if __name__ == "__main__":
    print("wrote", speak("Take your time. There is no rush at all.", emotion="warm"))

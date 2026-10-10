"""Use Sonify 1026 as a LangChain tool.

    pip install langchain-core "sonify1026[mcp]"

The tool returns the path of the generated WAV, so an agent can attach or
play it. Drop it into any LangChain agent's `tools=[...]` list.
"""

from __future__ import annotations

from langchain_core.tools import tool

from snfy.mcp_server import synthesize_to_file


@tool
def speak(text: str, emotion: str = "neutral", language: str = "auto") -> str:
    """Speak text aloud with an expressive Indian female voice.

    Saves the audio to a WAV file and returns its absolute path. Reads
    English, Hindi and Hinglish.

    Args:
        text: The words to speak.
        emotion: A Sonify preset, e.g. "warm", "happy", "calm", "whisper".
        language: "auto" (default), "english", "hindi" or "us".
    """
    return synthesize_to_file(text, emotion=emotion, language=language)["path"]


if __name__ == "__main__":
    path = speak.invoke(
        {"text": "Yaar, kal ki meeting bahut important hai.", "emotion": "warm"}
    )
    print("wrote", path)

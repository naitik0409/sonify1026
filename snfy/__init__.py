"""
Sonify 1026 Voice Engine
An expressive, adaptive Indian female TTS performance layer over Microsoft
Edge TTS, with emotion, accent, prosody and EN/HI code-switching control.
The neural voice and audio come from Edge TTS; Sonify does not ship its own
speech model — it directs and post-processes that voice.
"""

from .engine import (
    SonifyEngine,
    BIOS02VoiceEngine,
    Emotion,
    Language,
    Persona,
    ProsodyParams,
    SynthesisResult,
    create_engine,
    render,
    render_arc,
    speak,
    clear_backend_cache,
    REGISTERS,
    DEFAULT_REGISTER,
)

__version__ = "0.2.0"
__all__ = [
    "SonifyEngine",
    "BIOS02VoiceEngine",
    "Emotion",
    "Language",
    "Persona",
    "ProsodyParams",
    "SynthesisResult",
    "create_engine",
    "render",
    "render_arc",
    "speak",
    "clear_backend_cache",
    "REGISTERS",
    "DEFAULT_REGISTER",
]
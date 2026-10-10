"""
Sonify 1026 Voice Engine
An expressive, adaptive Indian female TTS engine with emotion, accent,
prosody and EN/HI code-switching control.
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
    "REGISTERS",
    "DEFAULT_REGISTER",
]
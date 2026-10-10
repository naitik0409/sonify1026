"""Sonify 1026 - interactive voice demo (Hugging Face Space).

Type any English, Hindi or Hinglish text, pick an emotion, and listen.

Backed by the `sonify1026` package on PyPI. Audio is served by Microsoft
Edge TTS, so the Space needs outbound internet (it has it).
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr

import snfy

EMOTIONS = [e.value for e in snfy.Emotion]

LANGUAGES = {
    "Auto-detect": None,
    "Indian English": snfy.Language.EN_IN,
    "Hindi": snfy.Language.HI_IN,
    "General American": snfy.Language.EN_US,
}

_engine = None


def get_engine() -> "snfy.SonifyEngine":
    global _engine
    if _engine is None:
        _engine = snfy.SonifyEngine()
    return _engine


def speak(text: str, emotion: str, language: str, speed: float, pitch: float):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to speak.")

    engine = get_engine()
    engine.set_speed(float(speed))
    engine.set_pitch(float(pitch))

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    result = engine.synthesize(
        text.strip(),
        emotion=emotion,
        language=LANGUAGES.get(language),
        output_path=path,
    )
    emo = getattr(result.emotion, "value", result.emotion)
    lang = getattr(result.language, "value", result.language)
    status = f"**{result.duration:.1f}s** · emotion `{emo}` · language `{lang}`"
    return path, status


EXAMPLES = [
    ["That is wonderful news - I am so happy for you!", "happy", "Indian English", 1.0, 0.0],
    ["Take your time. There is no rush at all.", "warm", "Indian English", 1.0, 0.0],
    ["Yaar, kal ki meeting bahut important hai.", "neutral", "Auto-detect", 1.0, 0.0],
    ["\u0920\u0940\u0915 \u0939\u0948, \u091c\u094b \u0915\u0930\u094b \u092a\u0942\u0930\u0947 \u092e\u0928 \u0938\u0947 \u0915\u0930\u094b\u0964", "warm", "Hindi", 1.0, 0.0],
    ["Come closer, I will tell you a secret.", "whisper", "Indian English", 0.95, 0.0],
]

INTRO = """
# Sonify 1026

**An expressive, adaptive Indian female voice** - warm and mid-register, it
speaks Indian English and code-switches seamlessly into native Hindi, and it
*performs* meaning instead of just reading punctuation.

Type English, Hindi or Hinglish below, choose an emotion, and press **Speak**.

> No API key, no account. Runs on the open-source
> [`sonify1026`](https://pypi.org/project/sonify1026/) package
> (`pip install snfy`).
"""

with gr.Blocks(title="Sonify 1026", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="Text",
                value="Namaste! I can read English, \u0939\u093f\u0928\u094d\u0926\u0940 and Hinglish.",
                lines=3,
                placeholder="Type something to speak...",
            )
            with gr.Row():
                emotion = gr.Dropdown(
                    choices=EMOTIONS, value="warm", label="Emotion"
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGES), value="Auto-detect", label="Language"
                )
            with gr.Row():
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                pitch = gr.Slider(-20, 25, value=0, step=1, label="Pitch (Hz)")
            with gr.Row():
                btn = gr.Button("Speak", variant="primary", scale=2)
                clear = gr.Button("Clear", scale=1)
        with gr.Column(scale=2):
            audio = gr.Audio(label="Audio", type="filepath", autoplay=True)
            status = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=[text, emotion, language, speed, pitch])

    gr.Markdown(
        "Made with [Sonify 1026](https://github.com/naitik0409/sonify1026) by "
        "**Naitik** · MIT licensed."
    )

    btn.click(
        speak,
        inputs=[text, emotion, language, speed, pitch],
        outputs=[audio, status],
    )
    clear.click(
        lambda: ("", "warm", "Auto-detect", 1.0, 0.0, None, ""),
        outputs=[text, emotion, language, speed, pitch, audio, status],
    )


if __name__ == "__main__":
    demo.queue().launch()

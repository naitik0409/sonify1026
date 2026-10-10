"""Sonify 1026 - interactive voice demo (Streamlit Community Cloud).

Type English, Hindi or Hinglish, pick an emotion, and listen. Backed by the
`snfy` package that ships in this repository.

Deploy: share.streamlit.io -> New app -> repo `naitik0409/sonify1026`,
branch `main`, main file `streamlit_app.py`.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

import snfy

st.set_page_config(page_title="Sonify 1026", page_icon="\U0001F399\uFE0F", layout="centered")

EMOTIONS = [e.value for e in snfy.Emotion]
LANGUAGES = {
    "Auto-detect": None,
    "Indian English": snfy.Language.EN_IN,
    "Hindi": snfy.Language.HI_IN,
    "General American": snfy.Language.EN_US,
}

DEFAULT_TEXT = "Namaste! I can read English, \u0939\u093f\u0928\u094d\u0926\u0940 and Hinglish."

EXAMPLES = {
    "Happy": "That is wonderful news - I am so happy for you!",
    "Warm": "Take your time. There is no rush at all.",
    "Hinglish": "Yaar, kal ki meeting bahut important hai.",
    "Hindi": "\u0920\u0940\u0915 \u0939\u0948, \u091c\u094b \u0915\u0930\u094b \u092a\u0942\u0930\u0947 \u092e\u0928 \u0938\u0947 \u0915\u0930\u094b\u0964",
    "Whisper": "Come closer, I will tell you a secret.",
}


@st.cache_resource(show_spinner=False)
def get_engine() -> "snfy.SonifyEngine":
    return snfy.SonifyEngine()


def use_example(value: str) -> None:
    st.session_state.text = value


st.title("\U0001F399\uFE0F Sonify 1026")
st.markdown(
    "**An expressive, adaptive Indian female voice** - warm and mid-register, it "
    "speaks Indian English and code-switches seamlessly into native Hindi, and it "
    "*performs* meaning instead of just reading punctuation.\n\n"
    "Type English, Hindi or Hinglish, choose an emotion, and press **Speak**."
)

if "text" not in st.session_state:
    st.session_state.text = DEFAULT_TEXT

st.markdown("**Try an example**")
cols = st.columns(len(EXAMPLES))
for col, (label, value) in zip(cols, EXAMPLES.items()):
    col.button(label, on_click=use_example, args=(value,), use_container_width=True)

text = st.text_area("Text", key="text", height=100)

col1, col2 = st.columns(2)
emotion = col1.selectbox("Emotion", EMOTIONS, index=EMOTIONS.index("warm"))
language = col2.selectbox("Language", list(LANGUAGES))
col3, col4 = st.columns(2)
speed = col3.slider("Speed", 0.5, 2.0, 1.0, 0.05)
pitch = col4.slider("Pitch (Hz)", -20, 25, 0, 1)

if st.button("Speak", type="primary", use_container_width=True):
    if not text or not text.strip():
        st.warning("Please enter some text to speak.")
    else:
        with st.spinner("Rendering voice..."):
            engine = get_engine()
            engine.set_speed(float(speed))
            engine.set_pitch(float(pitch))
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            result = engine.synthesize(
                text.strip(),
                emotion=emotion,
                language=LANGUAGES[language],
                output_path=path,
            )
            with open(path, "rb") as fh:
                st.session_state.audio = fh.read()
            emo = getattr(result.emotion, "value", result.emotion)
            lang = getattr(result.language, "value", result.language)
            st.session_state.info = f"{result.duration:.1f}s \u00b7 emotion {emo} \u00b7 language {lang}"

if st.session_state.get("audio"):
    st.audio(st.session_state.audio, format="audio/wav")
    st.download_button(
        "Download WAV",
        st.session_state.audio,
        file_name="sonify.wav",
        mime="audio/wav",
    )
    if st.session_state.get("info"):
        st.caption(st.session_state.info)

st.divider()
st.caption(
    "Made with [Sonify 1026](https://github.com/naitik0409/sonify1026) by "
    "**Naitik** \u00b7 MIT licensed \u00b7 [PyPI](https://pypi.org/project/sonify1026/) "
    "(`pip install snfy`)."
)

# Sonify 1026 Voice Engine

[![PyPI version](https://img.shields.io/pypi/v/sonify1026.svg)](https://pypi.org/project/sonify1026/)
[![Python](https://img.shields.io/pypi/pyversions/sonify1026.svg)](https://pypi.org/project/sonify1026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An expressive, adaptive **Indian female TTS voice engine** — mid-20s, urban Indian
English with native-standard Hindi code-switching.

```bash
pip install sonify1026
```

```python
import snfy

engine = snfy.SonifyEngine()
engine.synthesize(
    "That is wonderful news, I am so happy for you!",
    emotion=snfy.Emotion.HAPPY,
    output_path="out.wav",
)
```

> Installed as **`sonify1026`**, imported as **`snfy`** — the same pattern as
> `pip install beautifulsoup4` / `import bs4`.

---

## Voice profile

| Aspect | Behaviour |
|---|---|
| Timbre | Warm, mid register with low-to-mid chest resonance; light, mid-high pitch |
| Delivery | Spontaneous and extemporaneous — no metronomic line-reading |
| Expressiveness | Emotional valence analysed per sentence: smiling inflection on positive phrases, acoustic softening on tender ones, subglottic lift on emphatic ones |
| Pacing | Non-linear — pauses, rate and breathing scale with the emotional subtext |
| Imperfections | Soft breath intakes, mouth clicks, micro-pauses, occasional vocal fry |
| Clarity | Phase-preserving EQ, no compression, no time-stretching in the output path |
| Uniqueness | Every call is its own performance (no fixed seed) |

---

## Features

- **Bilingual** — English (`en-IN-NeerjaNeural`) and Hindi (`hi-IN-SwaraNeural`), with automatic Hinglish code-switching in either direction.
- **17 emotions** — `neutral`, `happy`, `calm`, `warm`, `gentle`, `sad`, `excited`, `angry`, `fearful`, `thoughtful`, `playful`, `tired`, `proud`, `surprised`, `reassuring`, `serious`, `whisper`.
- **Adaptive prosody** — intensity words, sentence length and emotional valence drive pitch, rate, volume, brightness, warmth, breathing interval and pause length per sentence.
- **Emotional arc** — a script that builds in feeling sounds like it builds, instead of resetting each sentence.
- **Register control** — `normal`, `formal`, `casual` (casual is the default).
- **Number normalisation** — digits, decimals, times, ordinals and phone numbers are spoken naturally (both ASCII and Devanagari).
- **Pitch guard** — verifies the rendered voice sits in the intended register and re-renders if a phrase drifts too low or high.
- **Offline fallback** — pyttsx3 when the network is unavailable.
- **Fully configurable** — every voice parameter lives in one YAML file; nothing is hardcoded.

---

## Installation

```bash
pip install sonify1026
```

Requires Python 3.9+. An internet connection is required at synthesis time
(audio is served by Microsoft Edge TTS); no API key needed.

## Usage

### Python API

```python
import snfy

engine = snfy.SonifyEngine()

# English
engine.synthesize("Hello! How are you today?", emotion=snfy.Emotion.WARM,
                  output_path="en.wav")

# Hindi
engine.synthesize("नमस्ते! आपका दिन शुभ हो।", emotion=snfy.Emotion.WARM,
                  language=snfy.Language.HI_IN, output_path="hi.wav")

# Hinglish — detected automatically
engine.synthesize("Yaar, kal ki meeting bahut important hai, on time aa jaana.",
                  output_path="hinglish.wav")
```

Work with audio in memory:

```python
import soundfile as sf

result = engine.synthesize("Hello!")      # no output_path
sf.write("out.wav", result.audio, result.sample_rate)
print(result.duration, result.sample_rate)  # 4.91 48000
```

### Runtime controls

```python
engine.set_speed(1.15)        # 0.5 - 2.0
engine.set_pitch(0)           # -20 - +25 Hz
engine.set_volume(1.0)        # 0.0 - 1.5
engine.set_offline_mode(True) # pyttsx3 fallback
engine.reload_config()        # re-read voice_config.yaml
```

### CLI

```bash
sonify1026 "Hello! How are you?" --emotion warm --output hello.wav
sonify1026 "यह एक परीक्षण है।" -e calm -o test.wav
sonify1026 "Play this." --play
```

---

## Configuration

All voice behaviour lives in `snfy/voice_config.yaml`. Use your own file with:

```python
engine = SonifyEngine(config_path="my_voice.yaml")
```

| Key | Effect |
|---|---|
| `character.base_pitch` | overall register (Hz offset) |
| `pitch_guard` | auto-corrects register if a phrase drifts |
| `persona.warm_hz` / `warm_gain` | chest resonance |
| `persona.presence_gain` | high-frequency clarity |
| `voice_settings.<voice>.eq` | per-voice tone shaping |
| `emotions.<name>` | per-emotion rate / volume / pause / compression |
| `dynamic_prosody.valence_words` | what counts as positive / tender / negative |
| `breath` | breath intakes before phrases |
| `humanize.max_utterance_chars` | where takes are split |

---

## Testing

```bash
pytest tests -q              # 56 objective quality/regression checks
python tests/voice_audit.py  # full voice QA report across a corpus
```

Clarity, loudness and modulation are measured objectively from the rendered
waveform and judged relative to the same text rendered straight through
edge-tts, so regressions in clarity or naturalness are caught automatically.

---

## License

MIT — see [LICENSE](LICENSE).

Audio is generated by Microsoft Edge TTS; you are responsible for complying with
its terms of use.
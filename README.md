# Sonify 1026

[![PyPI](https://img.shields.io/pypi/v/sonify1026.svg)](https://pypi.org/project/sonify1026/)
[![Python](https://img.shields.io/pypi/pyversions/sonify1026.svg)](https://pypi.org/project/sonify1026/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An expressive, adaptive Indian female TTS voice engine** — warm, mid-register,
urban Indian English that code-switches seamlessly into native-standard Hindi.

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
> `pip install beautifulsoup4` → `import bs4`.

---

## Contents

- [Why Sonify](#why-sonify)
- [Installation](#installation)
- [Importing](#importing)
- [Quickstart](#quickstart)
- [Emotions](#emotions)
- [Languages and code-switching](#languages-and-code-switching)
- [Runtime controls](#runtime-controls)
- [Batch and streaming](#batch-and-streaming)
- [Command line](#command-line)
- [Fine-tuning the voice](#fine-tuning-the-voice)
- [Testing](#testing)
- [Requirements and limits](#requirements-and-limits)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Why Sonify

Most TTS reads. Sonify **performs**:

- **Adapts to meaning, not just punctuation.** Intensity words, sentence length
  and emotional valence are analysed per sentence and drive pitch, rate,
  volume, brightness, warmth, breathing and pause length.
- **Emotional arc.** A script that builds in feeling sounds like it builds,
  instead of resetting to neutral on every sentence.
- **Natural imperfections.** Breath intakes, mouth clicks, micro-pauses and
  occasional vocal fry — the things that make speech sound human.
- **Subglottic effort.** Content words are projected, hedging words (*maybe*,
  *perhaps*) are delivered more softly.
- **Unique every call.** No fixed seed, so no two renders are identical.
- **Clarity first.** Phase-preserving EQ, no output compression, and no
  time-stretching in the signal path — so it stays intelligible.
- **Register safety.** A pitch guard verifies each render sits in the intended
  register and re-renders if a phrase drifts too low or high.

## Installation

```bash
pip install sonify1026
```

Python 3.9+. No API key, no account. Dependencies (`edge-tts`, `numpy`,
`scipy`, `soundfile`, `PyYAML`) install automatically.

From source:

```bash
pip install git+https://github.com/naitik0409/sonify1026.git
```

## Importing

Three equivalent forms — all verified working:

```python
import snfy                                     # module namespace (recommended)
from snfy import SonifyEngine, Emotion          # direct names
from snfy.engine import SonifyEngine            # from the module directly
```

```python
engine = snfy.create_engine()                   # factory
engine = snfy.SonifyEngine()                    # class
engine = snfy.BIOS02VoiceEngine()               # legacy alias, still supported
```

Everything public is re-exported from the top-level package:

```python
snfy.SonifyEngine, snfy.Emotion, snfy.Language, snfy.Persona,
snfy.ProsodyParams, snfy.SynthesisResult, snfy.create_engine,
snfy.render, snfy.render_arc, snfy.speak, snfy.REGISTERS,
```

> `import snfy` prints one `INFO` line on first use. That is the library
> starting up — silence it with `logging.getLogger("sonify1026").setLevel(logging.WARNING)`.

## Quickstart

```python
import snfy

engine = snfy.SonifyEngine()

# Write to a file
result = engine.synthesize("Hello! How are you?", output_path="hello.wav")
print(result.duration, "seconds")

# Or keep it in memory (float32 numpy array, 48 kHz)
import soundfile as sf
result = engine.synthesize("Hello again!")
sf.write("hello.wav", result.audio, result.sample_rate)

# Directories are created automatically
engine.synthesize("Nested path works too.", output_path="a/b/c/deep.wav")
```

`SynthesisResult` carries: `audio`, `sample_rate`, `duration`, `emotion`,
`language`, `text`.

## Emotions

17 presets, each shaping rate, volume, pause and tone:

| | | | |
|---|---|---|---|
| `neutral` | `happy` | `sad` | `angry` |
| `excited` | `calm` | `whisper` | `fearful` |
| `warm` | `gentle` | `thoughtful` | `playful` |
| `tired` | `proud` | `surprised` | `reassuring` |
| `serious` | | | |

```python
engine.synthesize("I am so happy for you!", emotion=snfy.Emotion.HAPPY)
engine.synthesize("I am here with you.", emotion=snfy.Emotion.REASSURING)
```

Emotion sets the *baseline*; the adaptive analysis still modulates each
sentence within it. Prefer plain strings too — `emotion="happy"` works.

## Languages and code-switching

```python
snfy.Language.EN_IN   # Indian English  (default)
snfy.Language.HI_IN   # Hindi
snfy.Language.EN_US   # General American
```

**You rarely need to specify it.** The engine detects Devanagari and switches
voices automatically, so mixed text just works:

```python
engine.synthesize("Yaar, kal ki meeting bahut important hai.")          # Hinglish
engine.synthesize("ठीक है, the point is simple — जो करो, पूरे मन से करो।")  # script mix
engine.synthesize("The order of 250 units ships on 12/05/2026.")       # numbers read naturally
```

## Runtime controls

```python
engine.set_speed(1.15)      # 0.5 – 2.0
engine.set_pitch(0)         # -20 – +25 Hz
engine.set_volume(1.0)      # 0.0 – 1.5
engine.set_offline_mode()   # pyttsx3 fallback when the network is down
```

Controls are per-engine and persist until changed.

## Batch and streaming

```python
# Many lines at once
results = engine.batch_synthesize(
    ["Good morning.", "How are you?", "See you soon."],
    [snfy.Emotion.WARM, snfy.Emotion.CALM, snfy.Emotion.HAPPY],
    output_dir="outputs/greetings",
)

# Async stream: yields (audio, sample_rate, pause_ms) per segment
import asyncio
async def speak():
    async for audio, sr, pause_ms in engine.stream("A longer sentence."):
        play(audio, sr)      # your audio device
        await asyncio.sleep(pause_ms / 1000)
asyncio.run(speak())
```

## Command line

```bash
snfy "Hello! How are you?" --emotion warm --output hello.wav
sonify1026 "यह एक परीक्षण है।" -e calm -o test.wav
snfy "Play this." --play          # needs: pip install sounddevice
```

| Flag | Meaning |
|---|---|
| `text` | what to say |
| `-e, --emotion` | any of the 17 presets (default `neutral`) |
| `-a, --accent` | `in` or `us` |
| `-o, --output` | output `.wav` path |
| `-p, --play` | play after synthesising |

## Fine-tuning the voice

Every voice parameter lives in one YAML file — nothing is hardcoded. Copy the
shipped config, edit it, and point the engine at your copy:

```python
engine = snfy.create_engine(config_path="my_voice.yaml")
engine.reload_config()          # or reload after editing at runtime
```

The most useful knobs:

| Key | Effect |
|---|---|
| `character.base_pitch` | overall register (Hz offset) |
| `persona.warm_hz` / `warm_gain` | chest resonance / body |
| `persona.presence_gain` | high-frequency clarity |
| `persona.target_rms_db` | loudness |
| `voice_settings.<voice>.eq` | per-voice tone shaping |
| `emotions.<name>` | per-emotion rate / volume / pause / compression |
| `dynamic_prosody.valence_words` | what counts as positive / tender / negative |
| `dynamic_prosody.intensity_volume_scale` | how strongly effort is projected |
| `breath` | breath intakes before phrases |
| `vocal_fry` | creak at phrase endings |
| `humanize.max_utterance_chars` | where long text is split |
| `registers` / `default_register` | `normal`, `formal`, `casual` |
| `pitch_guard` | register safety net |

Full reference: **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

## Testing

```bash
pytest tests -q              # 56 objective checks (needs a network)
python tests/voice_audit.py  # full voice QA report
```

## Requirements and limits

- **Requires an internet connection** at synthesis time — audio is served by
  Microsoft Edge TTS. `set_offline_mode()` falls back to pyttsx3, with much
  lower quality.
- Edge TTS free endpoints are provided as-is; for production, read and comply
  with Microsoft's terms of use.
- First call takes 1–3 seconds (network round trip).
- Devanagari text prints as `?` in a default Windows console — that is the
  terminal's encoding, not the audio. Set `PYTHONIOENCODING=utf-8` if you need
  to print it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NoAudioReceived` | Text produced no speech — check it isn't empty or punctuation-only |
| Connection errors | No internet, or a proxy; retry, or use `set_offline_mode()` |
| `speak()` / `--play` fails | `pip install sounddevice` |
| Unwanted `outputs/temp_synthesis.wav` | `synthesize()` always writes a file; pass `output_path=` to control it |
| Hindi prints as `?` | Console encoding; set `PYTHONIOENCODING=utf-8` |
| Want the same delivery twice | Pass `seed=12345` (defaults to a fresh random delivery) |

More: **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**.

## Documentation

| Guide | Contents |
|---|---|
| [docs/API.md](docs/API.md) | Complete API reference |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every config key, with tuning guidance |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Problems and fixes |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## License

MIT — see [LICENSE](LICENSE).
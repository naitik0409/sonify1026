# API Reference — `snfy`

Complete reference for the `snfy` package (distribution `sonify1026`).

---

## Module exports

```python
import snfy
snfy.__version__          # e.g. "0.1.0"
```

| Name | Kind | Purpose |
|---|---|---|
| `SonifyEngine` | class | Main engine. Alias: `BIOS02VoiceEngine` |
| `create_engine(config_path=None)` | function | Convenience factory |
| `Emotion` | enum | 17 emotion presets |
| `Language` | enum | `EN_IN`, `HI_IN`, `EN_US` |
| `Persona` | dataclass | Voice character settings |
| `ProsodyParams` | dataclass | Prosody bundle |
| `SynthesisResult` | dataclass | What synthesis returns |
| `render(...)` | function | Low-level single-utterance render |
| `render_arc(...)` | function | Low-level multi-step render |
| `speak(...)` | function | Render and play immediately |
| `REGISTERS`, `DEFAULT_REGISTER` | dict/str | Delivery registers |

---

## `SonifyEngine`

```python
engine = snfy.SonifyEngine(config_path: str | None = None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `config_path` | `str \| None` | `None` | Path to a custom voice YAML. `None` uses the packaged config |

Attributes: `config` (dict), `config_path` (str), `persona` (`Persona`),
`sample_rate` (int).

---

### `synthesize(...)` → `SynthesisResult`

```python
engine.synthesize(
    text: str,
    emotion: Emotion | str = Emotion.NEUTRAL,
    language: Language | None = None,
    output_path: str | None = None,
    accent: str = "in",
    register: str = "normal",
    seed: int | None = None,
) -> SynthesisResult
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `text` | `str` | — | Text to speak. Any length; long text is split at sentence boundaries |
| `emotion` | `Emotion \| str` | `neutral` | Emotion preset; strings accepted |
| `language` | `Language \| None` | auto | `None` detects Hindi by script and picks the voice for you |
| `output_path` | `str \| None` | `None` | Destination `.wav`. Missing directories are created. If `None`, a temp file is written to `outputs/temp_synthesis.wav` **and** the audio is returned |
| `accent` | `str` | `"in"` | `"in"` (Indian English) or `"us"` |
| `register` | `str` | `"normal"` | `normal`, `formal`, or `casual` |
| `seed` | `int \| None` | `None` | `None` = a new random delivery every call. Pass an integer to pin the performance decisions |

**Behaviour notes**

- Every call produces a *different* performance unless `seed` is given.
- Raises if the backend returns no audio (e.g. empty or punctuation-only text).
- Returns the audio in memory whether or not `output_path` is supplied.

---

### `batch_synthesize(...)` → `list[SynthesisResult]`

```python
engine.batch_synthesize(
    texts: list[str],
    emotions: list[Emotion] | None = None,
    output_dir: str = "outputs",
) -> list[SynthesisResult]
```

Synthesises many lines in one call. `emotions` may be shorter than `texts`, in
which case the remaining lines use the neutral default. Each result corresponds
to one input line, in order.

```python
results = engine.batch_synthesize(
    ["Good morning.", "How are you?"],
    [snfy.Emotion.WARM, snfy.Emotion.CALM],
    output_dir="outputs/greetings",
)
```

---

### `stream(...)` → async generator

```python
engine.stream(
    text: str,
    emotion: Emotion = Emotion.NEUTRAL,
    accent: str = "in",
)
```

Yields one tuple per synthesised segment:

```python
(audio: numpy.ndarray, sample_rate: int, pause_ms: float)
```

- `audio` — `float32` array for that segment only
- `pause_ms` — silence to insert after the segment before playing the next

```python
import asyncio, sounddevice as sd, snfy

async def speak(text):
    async for audio, sr, pause_ms in snfy.create_engine().stream(text):
        sd.play(audio, sr)
        sd.wait()
        await asyncio.sleep(pause_ms / 1000)

asyncio.run(speak("A long paragraph spoken segment by segment."))
```

> `stream` is an **async generator** — iterate it inside `async def` only.

---

### `speak(...)`

```python
engine.speak(
    text: str,
    emotion: Emotion = Emotion.NEUTRAL,
    language: Language | None = None,
)
```

Synthesises and plays immediately. Requires `sounddevice`
(`pip install sounddevice`). Returns nothing.

---

### Controls

| Method | Range | Effect |
|---|---|---|
| `engine.set_speed(speed)` | `0.5 – 2.0` | Speaking rate for subsequent calls |
| `engine.set_pitch(hz)` | `-20 – +25` | Pitch offset in Hz |
| `engine.set_volume(v)` | `0.0 – 1.5` | Linear volume scale |
| `engine.set_offline_mode(enabled=True)` | bool | pyttsx3 fallback when offline |
| `engine.reload_config(config_path=None)` | `str \| None` | Re-read the YAML; optional new path |

Values persist on the instance until changed. `reload_config()` with no
argument re-reads the current file — useful for live tuning:

```python
engine = snfy.create_engine(config_path="my_voice.yaml")
# ... edit my_voice.yaml in another window ...
engine.reload_config()
```

---

## `SynthesisResult`

| Field | Type | Meaning |
|---|---|---|
| `audio` | `numpy.ndarray` (float32) | Mono waveform |
| `sample_rate` | `int` | Samples per second (`48000`) |
| `duration` | `float` | Seconds |
| `emotion` | `Emotion` | Emotion used |
| `language` | `Language` | Language resolved |
| `text` | `str` | Input text |

```python
import soundfile as sf
r = engine.synthesize("Save me.")
sf.write("out.wav", r.audio, r.sample_rate)
```

---

## Enums

### `Emotion`

```
neutral   happy     sad       angry     excited
calm      whisper   fearful   warm      gentle
thoughtful playful  tired     proud     surprised
reassuring serious
```

### `Language`

```
EN_IN   # Indian English (default)
HI_IN   # Hindi
EN_US   # General American
```

---

## `Persona` and `ProsodyParams`

```python
snfy.Persona()        # base_pitch, highpass_hz, warm_hz, warm_gain,
                      # pres_hz, pres_gain, target_rms_db, peak_ceiling_db
snfy.ProsodyParams()  # pitch_shift, rate_scale, energy, pause_factor
```

`Persona` is populated from the `persona` and `character` config sections when
the engine starts; adjust those in YAML rather than editing the dataclass.

---

## Low-level functions

For advanced use — `SonifyEngine` is a thin, safer wrapper.

```python
snfy.render(text, out_wav, emotion="neutral", persona=None,
            accent="in", seed=20261009, breaths=True,
            config=None, register="normal")

snfy.render_arc(steps, out_wav, persona=None, accent="in", seed=20261009,
                breaths=True, config=None, register="normal")

snfy.speak(text, emotion="neutral", accent="in")
```

`render_arc` takes `steps` as a list of `(text, emotion)` pairs and renders
them as one continuous track with crossfades — useful for multi-line delivery.

These accept the **raw config dict**, so pass `engine.config` if you want the
tuned settings:

```python
snfy.render("Hello.", "out.wav", config=snfy.create_engine().config)
```

---

## Logging

The library logs one startup line under the `sonify1026` logger. Silence it:

```python
import logging
logging.getLogger("sonify1026").setLevel(logging.WARNING)
```
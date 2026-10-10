# Troubleshooting

## Installation & import

**`ModuleNotFoundError: No module named 'snfy'`**
The package installs as `sonify1026` (or the alias `snfy`) but imports as `snfy`:
```python
import snfy          # correct
import sonify1026    # wrong - that is the pip name, not the module
```
Install with either name: `pip install sonify1026` or `pip install snfy`.

**Wrong version installed, or an old one is picked up**
```bash
pip uninstall -y sonify1026 snfy
pip install --upgrade sonify1026     # or: pip install --upgrade snfy
python -c "import snfy; print(snfy.__file__, snfy.__version__)"
```
If you also have the project checked out locally, running from that folder
loads the local copy instead of the installed one. Check with `snfy.__file__`.

**Import is slow / hangs on first use**
Normal — the engine reads and parses its YAML config on construction. The
first synthesis additionally makes a network round trip.

---

## Runtime errors

**`edge_tts.exceptions.NoAudioReceived`**
The backend returned no speech. Usual causes: empty text, whitespace, or
punctuation-only input (`"..."`, `"…"`). Guard your input:
```python
if not text.strip():
    return
```

**Connection / timeout errors**
Synthesis is a network call. Check connectivity, proxy settings, and retry.
For a degraded offline fallback:
```python
engine.set_offline_mode(True)   # swaps to the pyttsx3 backend
```
This is a **different, OS-provided voice**, not the Sonify/Edge voice — much
lower quality, a safety net rather than the same sound.

**`ImportError: No module named 'sounddevice'`**
Only needed for `engine.speak()` and `snfy "text" --play`:
```bash
pip install sounddevice
```
Synthesis to a file never needs it.

**`OSError` / `PermissionError` writing output**
Point `output_path` somewhere writable. Note `synthesize()` writes a file even
when you only want the audio in memory (see below).

**`asyncio` errors around `stream()`**
`stream` is an **async generator** — iterate it inside `async def`, never
synchronously:
```python
async def go():
    async for audio, sr, pause_ms in engine.stream(text):
        ...
```

---

## Unexpected files

**`outputs/temp_synthesis.wav` keeps appearing**
`synthesize()` always renders to disk. If you call it without `output_path` it
uses that temp path, then returns the array too. To keep your project clean,
either pass an explicit path, or write the returned audio yourself:
```python
import soundfile as sf
r = engine.synthesize(text)
sf.write("out.wav", r.audio, r.sample_rate)
```

---

## Console / terminal

**Hindi text prints as `?` or `??`**
The Windows console is not UTF-8 — the **audio is unaffected**. Either don't
print it, or:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```
or in Python: `sys.stdout.reconfigure(encoding="utf-8")`.

**The `INFO:sonify1026: ... initialized` line**
That is the library starting up, not an error. Silence it:
```python
import logging
logging.getLogger("sonify1026").setLevel(logging.WARNING)
```

---

## Voice quality

**Sounds unclear / muffled**
1. Raise `persona.pres_gain` (clarity) and `persona.presence_gain` (air)
2. Lower `breath.level_normal` — breath noise veils consonants
3. Keep `pitch_contour` and `word_elongation` **disabled**; both are
   time-domain operations that blur consonants
4. Verify with `python tests/voice_audit.py`

**Sounds too thin or small**
Raise `persona.warm_gain` (0.65 → 0.9) and add low-mid EQ in
`voice_settings.<voice>.eq`; lower `character.base_pitch`.

**Sounds too deep / reads as male**
Raise `character.base_pitch` and check `pitch_guard.min_f0_hz`. Measure:
```bash
python tests/voice_audit.py      # reports F0 for every case
```

**Sounds robotic or monotonous**
- Raise `variation.rate_pct` and `pitch_hz`
- Add `humanize.max_pitch_step_hz` smoothing (lower = smoother joins)
- Enable `micro_prosody`, `word_life`, `phrase_swell`
- Add words to `dynamic_prosody.intensity_words` so real content triggers effort

**A stray "ahm" at the end of the sentence**
`pause_filler` speaks hesitation words. It is off by default; if you enabled
it, turn it off:
```yaml
pause_filler: {enabled: false}
```

**Shivering / wavering between sentences**
- Set `humanize.max_pitch_step_hz` to `2` or `3`
- Keep `pitch_contour.enabled: false` and `word_elongation.enabled: false`
- Raise `humanize.soft_max_utterance_chars` so text splits at sentence ends

**A word sounds repeated at a phrase boundary**
Indicates a take was cut mid-sentence. Raise `humanize.soft_max_utterance_chars`
(default 520). Takes should only ever end on `.`, `!`, `?`, `…` or `।`.

---

## Reproducibility

**Need the same delivery twice?**
Pass a seed — each call is otherwise intentionally unique:
```python
engine.synthesize(text, seed=12345)
engine.synthesize(text, seed=12345)
```
Prosody decisions then match. Waveform samples may still differ slightly,
because the upstream speech service (Edge TTS) is not bit-deterministic.

**Need byte-identical output?**
Enable the raw-audio cache, which reuses the exact neural audio for identical
takes, so a fixed `seed` becomes fully reproducible (and network calls are saved):
```yaml
backend_cache:
  enabled: true
```
```python
import snfy.engine
snfy.engine.clear_backend_cache()   # start fresh if needed
```
The cache is in-process only and off by default.

---

## Performance

| Symptom | Action |
|---|---|
| First call is slow | Network round trip; subsequent calls are faster |
| Long text is slow | Lower `humanize.max_utterance_chars` (more, smaller segments) |
| Memory heavy on long text | Lower `output.sample_rate` to `24000` |

---

## Still stuck?

Include: the command or code, the full traceback, your config path if you
changed it, and the output of:

```bash
python -c "import snfy; print(snfy.__version__, snfy.__file__)"
```
# Benchmarking — emotion differentiation & speaker identity

`tests/emotion_benchmark.py` is the objective acceptance harness for the two
assessment priorities that need measurement before tuning can be verified:

- **P6 / P3 — emotion differentiation**: do the shipped emotion presets change
  the acoustic output in the direction the config claims, in *both* English
  and Hindi?
- **P6 — cross-language speaker identity**: does `voice_mode: single` really
  hold *one* voice register across Latin and Devanagari text?

Everything is dependency-free (stdlib `wave`/`array`/`math` + NumPy-style
lists from `tests/metrics.py`), deterministic (fixed seed `20261102`), and
asserted in CI-style thresholds — not by ear.

## Running it

```bash
python tests/emotion_benchmark.py            # full matrix (17 emotions x EN+HI)
python tests/emotion_benchmark.py --quick    # 7-emotion subset (~1 min)
python tests/emotion_benchmark.py --reuse    # re-analyse existing WAVs, no network
python tests/emotion_benchmark.py --lang en  # one language
python tests/emotion_benchmark.py --json     # machine-readable report

pytest tests/test_emotion_benchmark.py -q -m "not network"   # offline units
pytest tests/test_emotion_benchmark.py -q -m network         # live assertions
```

Artifacts land in `outputs/benchmark/` (per-cell WAVs + `emotion_report.json`).
`--reuse` re-measures the WAVs already there — the fast loop for calibrating
thresholds without re-rendering (same seed only).

## What is measured

### Emotion differentiation (per language)

Each cell renders the **same neutral text** in one emotion through the shipped
preset, then `metrics.emotion_features()` extracts:

| feature      | acoustic cue                                            |
|--------------|---------------------------------------------------------|
| `f0_median`  | pitch level (arousal — Eyben 2010, Pereira 1998)         |
| `f0_span`    | pitch range / expressiveness                            |
| `pace`       | syllables/sec (arousal — tempo)                         |
| `dyn_db`     | energy dynamics                                         |
| `clarity`    | spectral clarity                                        |
| `centroid_hz`| spectral centroid (brightness)                          |

Deliberately **excluded**:

- `rms_db` — pinned at ~−20.5 dB by the mastering/normalisation stage, so it
  carries no emotion information (it is computed and reported, just not scored).
- `rate_syll` — the syllable-nucleus counter is noisy on Hindi retroflexes;
  `pace` (1/duration of the voiced contour) is the reliable rate cue.

**Measured arousal** = mean z-score of `f0_median`, `f0_span`, `pace` across
cells of one language. Then:

- **Spearman rho** between the *config-implied* arousal (from
  `pitch`/`rate`/`volume`/`tone` deltas in `voice_config.yaml`) and the
  *measured* arousal — does the engine order emotions as configured?
- **Separation from neutral** — z-distance of each emotion from the neutral
  cell (neutral = 0 by construction).

### Speaker identity

| check | how | why |
|-------|-----|-----|
| F0 delta EN↔HI | median F0 of neutral EN vs HI | register must not jump across scripts (floor ≤ 30 Hz) |
| F0-IQR ratio | pitch-variability stability | ≤ 2.0 |
| routing | `voice_mode: single` resolves both texts to one voice key | asserted directly |
| MCD two-speaker control | same text, Swara vs Neerja (dual mode) | must be **high** (≥ 4 dB) — proves the metric detects speaker changes |
| MCD same-voice control | same text, same voice, two renders | must be **low** (≤ 2 dB) — proves routing is stable |
| MCD cross-language | EN vs HI neutral | **informational only** — MCD is phonetic-content confounded (EN↔HI ≈ 9.2 dB ≈ two-speaker same-text ≈ 9.9 dB), so it is reported but never asserted |

MCD = mel-cepstral distortion over 26-coefficient MFCCs (`metrics.mcd`).

## Thresholds

Defined in `THRESHOLDS` (`tests/emotion_benchmark.py`), split per language
because Hindi delivery has a genuinely lower arousal spread today (Priority
2/3 backlog) — the bar is honest, not equal:

| key | value | rationale |
|-----|-------|-----------|
| `min_spearman_rho_en` | 0.45 | English must track config ordering clearly |
| `min_spearman_rho_hi` | 0.30 | Hindi floor — documented gap, tighten as P2/P3 lands |
| `min_mean_separation` | 0.60 z | renders must not be interchangeable |
| `min_extreme_separation_en` | 0.45 z | happy/excited/angry/sad/calm/whisper vs neutral |
| `min_extreme_separation_hi` | 0.35 z | same, lower bar |
| `max_f0_delta_hz` | 30 | one register across EN/HI |
| `max_f0_iqr_ratio` | 2.0 | pitch variability stable |
| `min_mcd_two_speakers_db` | 4.0 | metric discriminates speakers |
| `max_mcd_same_voice_text_db` | 2.0 | routing stable |

## Current baseline (v0.3.0, edge-tts 7.3.1, seed 20261102)

Full 17-emotion matrix, **21/21 checks pass** (~110 s):

| | EN | HI |
|---|---|---|
| Spearman rho | +0.86 (p=0.0002) | +0.79 (p=0.0002) |
| mean separation | 2.82 z | 2.29 z |
| extremes (min) | sad 6.03 z | happy 1.33 z |

Identity: F0 delta 2.7 Hz, IQR ratio 1.07, routing EN→swara HI→swara,
two-speaker MCD 9.82 dB vs same-voice 0.00 dB.

The quick subset measures EN ρ +0.96 / HI ρ +0.86. Note: an earlier
exploratory run showed HI ρ −0.71 — that was a *measurement* artifact
(RMS pinned by mastering + noisy syllable counter on Hindi), not a broken
Hindi voice; the current feature set fixes it.

## Interpreting a failure

- **rho fails, separations pass** → emotions differ, but not in the *order*
  the config claims: check `voice_config.yaml` pitch/rate deltas for the
  offending presets.
- **separations fail** → presets are acoustically interchangeable: the
  emotion DSP (pitch_guard, rate, tone) is not biting.
- **F0 delta fails** → `voice_mode: single` routing broke or one script's
  text is hitting a different voice path.
- **two-speaker control fails (MCD too low)** → the identity metric lost
  discrimination — distrust all identity numbers until fixed.

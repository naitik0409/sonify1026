# Configuration Reference — `snfy`

Every aspect of the voice is controlled by YAML. Nothing is hardcoded in the
engine, so you can retune the entire voice without touching code.

## Using a custom config

```python
import snfy, shutil, pathlib

# 1. Start from the shipped config
src = pathlib.Path(snfy.__file__).parent / "voice_config.yaml"
shutil.copy(src, "my_voice.yaml")

# 2. Edit my_voice.yaml — any key you like
# 3. Load it
engine = snfy.create_engine(config_path="my_voice.yaml")

# 4. Or swap it on a running engine
engine.reload_config("my_voice.yaml")
```

Values are validated on load; unknown keys are ignored, so you can keep notes
in the file.

---

## `character`

Identity and global register.

| Key | Default | Meaning |
|---|---|---|
| `name` | `"Sonify 1026"` | Display name of the voice |
| `base_pitch` | `-5` | **Overall register, in Hz.** More negative = deeper. This is the single biggest lever on perceived voice |
| `pitch_min` / `pitch_max` | `-45` / `25` | Clamp range for computed pitch |
| `speaking_rate` | `1.0` | Global rate multiplier |
| `default_emotion` | `neutral` | Emotion used when none is given |
| `randomize_seed` | `true` | `true` = every call is a unique performance. Set `false` (+ `seed`) for reproducible output |

> `base_pitch` interacts with `pitch_guard` below. Raising it makes the voice
> lighter and younger; lowering it makes it deeper. Keep female identity by
> letting the guard work rather than guessing.

## `output`

| Key | Default | Meaning |
|---|---|---|
| `sample_rate` | `48000` | Output sample rate. 24000 halves file size |

## `emotions`

Per-emotion prosody. **Set `comp: 1.0` to disable compression** (already the
default throughout).

| Key | Meaning |
|---|---|
| `pitch` | Pitch offset in Hz, added to the sentence's computed pitch |
| `rate` | Rate string, e.g. `"-4%"` |
| `volume` | Volume string, e.g. `"+2%"` |
| `pause` | Pause multiplier — `1.2` = 20% longer between sentences |
| `bright` | Presence/EQ brightness |
| `warm` | Chest-resonance EQ |
| `dark` | Optional low-pass cutoff (Hz); `null` disables |
| `comp` | Compression ratio — **`1.0` = off** |
| `air` | Room-air level (only when `room_tone` is enabled) |

Available: `neutral`, `warm`, `happy`, `calm`, `gentle`, `sad`, `excited`,
`whisper`, `angry`, `fearful`, `thoughtful`, `playful`, `tired`, `proud`,
`surprised`, `reassuring`, `serious`.

**Tuning for a softer, warmer persona:**
```yaml
emotions:
  neutral: {pitch: 0, rate: "-4%", volume: "-1%", pause: 1.06,
            bright: 0.15, warm: 0.5, dark: null, comp: 1.0, air: -47}
```

## `voices` and `accents`

```yaml
voices:
  neerja: {id: "en-IN-NeerjaNeural", pitch: 0, lo: -45, hi: 25}
  swara:  {id: "hi-IN-SwaraNeural",  pitch: 0, lo: -45, hi: 25}
  aria:   {id: "en-US-AriaNeural",   pitch: 0, lo: -45, hi: 25}

accents:
  in: {voice: "neerja", rate: -2, label: "Indian English"}
  us: {voice: "aria",   rate: 0,  label: "General American"}
default_accent: "in"
```

Any Edge TTS voice ID works. `pitch` is an extra per-voice offset; `lo`/`hi`
bound it. **Keep `lo` at or below `character.base_pitch`** or the offset will be
clipped away.

## `persona`

The DSP character of the voice.

| Key | Default | Meaning |
|---|---|---|
| `highpass_hz` | `95.0` | Rumble filter. Raise to thin the voice, lower for more body |
| `warm_hz` | `175.0` | Centre frequency of the chest-resonance boost |
| `warm_gain` | `0.65` | Chest-resonance depth. **The warmth control** |
| `pres_hz` | `3400.0` | Presence frequency (articulation) |
| `pres_gain` | `1.3` | Presence depth — raise for clearer consonants |
| `target_rms_db` | `-20.5` | Loudness target |
| `peak_ceiling_db` | `-1.0` | Peak ceiling. Enforced after resampling |
| `presence_hz` | `7800.0` | Air frequency |
| `presence_gain` | `1.6` | Air / brightness. Raise for a brighter, younger voice |

> If the voice sounds dull or unclear, raise `pres_gain` and `presence_gain`
> before touching anything else. If it sounds harsh, lower them.

## `puncturation`

| Key | Default | Meaning |
|---|---|---|
| `sentence_pause_ms` | `240` | Base pause between sentences |
| `pause_jitter` | `0.30` | Random ±30% on every pause. `0` = robotic; raise for more drift |
| `sentence` / `question` / `exclaim` / `clause` / `ellipsis` | — | Per-type `pause_ms`, plus `pitch_drop` / `pitch_rise` / `energy_boost` |
| `cognitive_pause` | — | `{probability, min_factor, max_factor}` — occasional long "thinking" pauses |

## `humanize`

| Key | Default | Meaning |
|---|---|---|
| `max_utterance_chars` | `260` | Soft target for splitting long text into takes |
| `soft_max_utterance_chars` | `520` | Hard limit — a take is never cut mid-sentence |
| `contractions` | `true` | Expand contractions before speaking |
| `lead_ms` / `tail_ms` | `40` / `110` | Silence padding at the start/end |
| `crossfade_ms` | `28` | Overlap when joining takes |
| `max_pitch_step_hz` | `3` | Maximum pitch jump between takes. Lower = smoother joins |

## `pitch_guard`

Verifies the rendered voice and re-renders if it drifts out of register.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Turn the safety net on |
| `min_f0_hz` | `196.0` | Below this the voice can read as male |
| `max_f0_hz` | `288.0` | Above this it strains |
| `max_attempts` | `2` | Re-render at most this many times |
| `max_adjust_hz` | `12.0` | Pitch nudge per attempt |
| `sample_seconds` | `6.0` | How much audio to measure |

## `prosody`

| Key | Default | Meaning |
|---|---|---|
| `enable_dynamic_pitch` | `true` | Apply computed pitch shifts |
| `enable_dynamic_rate` | `true` | Apply computed rate shifts |
| `enable_energy_control` | `true` | Apply computed volume shifts |
| `sentence_level_variation` | `0.08` | Sentence-to-sentence variation |
| `word_level_variation` | `0.02` | Word-to-word variation |

## `dynamic_prosody`

The adaptive brain. **This is the most important section for expressiveness.**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch |
| `intensity_words` | — | `{high, medium, low}` word lists (English + Hindi). Drives effort |
| `intensity_pitch_scale` | `0.8` | How much intensity lifts pitch |
| `intensity_rate_scale` | `0.7` | How much intensity speeds/slows delivery |
| `intensity_volume_scale` | `5.0` | **Subglottic effort** — how strongly key content projects |
| `short_sentence_chars` / `long_sentence_chars` | `25` / `75` | Length thresholds for the rate nudge |
| `length_rate_delta` | `2` | Magnitude of that nudge |
| `valence_words` | — | `{positive, tender, negative}` lists driving smile/softening |
| `smile_pitch_st` | `1.4` | Semitones of lift on positive text |
| `smile_brightness_db` | `1.2` | Brightening on positive text |
| `smile_volume_db` | `-1.2` | Softening on warm text |
| `smile_rate_pct` | `-1.5` | Warm text eases slightly |
| `soften_warm_db` | `0.5` | Extra warmth on tender text |
| `valence_pause_scale` | `0.14` | Emotional subtext lengthens/shortens pauses |
| `arc_pitch_st` / `arc_volume_db` | `0.9` / `2.0` | Emotional build across a passage |

**Add your own words** — this is the cheapest way to make the voice sound like
it understands your content:

```yaml
dynamic_prosody:
  intensity_words:
    high: [critical, urgent, launch, deploy, तय, ज़रूरी]
    medium: [update, review, समीक्षा]
    low: [maybe, whatever, eventually, शायद]
  valence_words:
    positive: [celebrate, win, शानदार]
    tender: [care, safe, प्यार]
    negative: [risk, delay, खतरा]
```

## `breath`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Breath intakes before phrases |
| `sentence_prob` / `question_prob` / `exclaim_prob` / `ellipsis_prob` / `clause_prob` | various | Chance of a breath per segment type |
| `default_prob` | `0.38` | Fallback |
| `emotional_boost` | `0.15` | Extra chance for soft emotions |
| `short_phrase_chars` / `short_phrase_prob_scale` | `45` / `0.4` | Short phrases breathe far less often |
| `duration_min` / `duration_max` | `0.12` / `0.20` | Breath length in seconds |
| `level_normal` / `level_emotional` | `0.065` / `0.09` | Breath level — **lower = cleaner speech** |

## `keyword_emphasis` and `keyword_breath`

| Key | Default | Meaning |
|---|---|---|
| `keyword_emphasis.enabled` | `true` | Emphasise content words |
| `keyword_emphasis.boost_db` | `2.0` | Emphasis depth |
| `keyword_emphasis.soften_db` | `-1.2` | Hedging words delivered more softly |
| `keyword_breath.enabled` | `true` | Small breath before an emphasised word |
| `keyword_breath.probability` | `0.14` | How often |

## Imperfection controls

All default to `enabled: false` except where noted — turn them on for a more
human, less polished delivery.

| Section | Key | Meaning |
|---|---|---|
| `micro_prosody` | `enabled` | **on** — `amplitude_shimmer_db` (0.25), `rate_hz` (6.0). Amplitude shimmer |
| `vocal_fry` | `enabled`, `probability` (0.45), `intensity` (0.11), `duration_frac`, `max_seconds`, `pulse_hz` | Creak at phrase endings |
| `mouth_click` | `enabled`, `probability` (0.18), `duration_ms`, `level_db` (-45) | Lip-parting transient |
| `pause_filler` | `enabled` | **Spoken hesitations ("uhm"). Off by default** — reads as a stray "ahm" at the end |
| `word_life` | `enabled`, `db` (1.4) | Word-level energy variation |
| `phrase_swell` | `enabled`, `db` (1.6) | Phrase-level amplitude contour |
| `room_tone` | `enabled`, `room_ratio`, `breath_ratio` | Room ambience |
| `air_breath` | `lowpass_hz`, `bandwidth`, `intensity_offset` | Breath-noise character |
| `coarticulation` | `enabled`, `window_ms` | Smoothing across segments |
| `pitch_contour` | `enabled` | **Off.** Time-domain pitch arc — it smears consonants |
| `word_elongation` | `enabled` | **Off.** Time-domain stretching — blurs consonants |

> Turning on `pitch_contour` or `word_elongation` will make the voice sound
> more "expressive" in a demo but noticeably blurrier. They are off for
> phonetic clarity; leave them off unless you have a specific reason.

## `registers` and `default_register`

| Key | Default | Meaning |
|---|---|---|
| `registers.normal` | rate 0 | Neutral delivery |
| `registers.formal` | rate −3, pitch −1 | Measured, slower, darker |
| `registers.casual` | rate +3, pitch +1 | Relaxed — **the default** |
| `default_register` | `casual` | Used when none is given |

## `variation`

| Key | Default | Meaning |
|---|---|---|
| `pitch_hz` | `1.0` | Random pitch drift per segment |
| `rate_pct` | `2.0` | Random rate drift per segment |
| `volume_db` | `0.6` | Random level drift per segment |
| `burst` | `{probability: 0.30, min_pct: 4, max_pct: 9}` | Occasional fast runs of words |

## `voice_settings`

Per-voice EQ and level.

```yaml
voice_settings:
  neerja:
    gain_db: 0.0
    eq:
      - {type: "pk", freq: 300.0,  gain: -0.4, q: 0.8}   # de-mud
      - {type: "pk", freq: 2800.0, gain: 1.6,  q: 0.9}   # presence
      - {type: "pk", freq: 9500.0, gain: 2.8,  q: 0.6}   # air
```

Filter types: `pk` (peaking), `hp` (high-pass), `lp` (low-pass).
Negative `gain` cuts. Apply to `neerja`, `swara`, `aria`.

## `sentence_type_rules`

Per-punctuation prosody adjustments.

```yaml
sentence_type_rules:
  sentence:  {pitch:  0, rate:  0, volume:  0}
  question:  {pitch:  3, rate: -1, volume:  0}
  exclaim:   {pitch:  4, rate:  2, volume:  2}
  ellipsis:  {pitch: -1, rate: -2, volume: -1}
  clause:    {pitch:  1, rate:  0, volume:  0}
  aside:     {pitch: -2, rate: -1, volume: -2}   # text in parentheses
```

## `emphasis`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Boost words wrapped in `**asterisks**` |
| `boost_db` | `1.8` | Emphasis depth |
| `ramp_factor` | `0.4` | Ramp sharpness |

---

## Recipes

**Deeper, more mature voice**
```yaml
character: {base_pitch: -15}
persona:   {warm_gain: 0.9}
pitch_guard: {min_f0_hz: 185.0}
```

**Younger, brighter, lighter**
```yaml
character: {base_pitch: 5}
persona:   {presence_gain: 2.0, warm_gain: 0.45}
```

**Warmer and softer for long-form reading**
```yaml
persona: {target_rms_db: -22.0, warm_gain: 0.8}
emotions:
  neutral: {rate: "-6%", pause: 1.15, warm: 0.6, comp: 1.0, ...}
breath:  {level_normal: 0.05}
```

**Maximum articulation / clearest speech**
```yaml
persona: {pres_gain: 1.8, presence_gain: 2.2, highpass_hz: 110.0}
breath: {level_normal: 0.04}
```
Then verify with `python tests/voice_audit.py`.

**More expressive, still clear**
```yaml
dynamic_prosody:
  intensity_volume_scale: 7.0
  smile_pitch_st: 1.8
  valence_pause_scale: 0.2
variation: {pitch_hz: 1.4, rate_pct: 2.4}
```

**Disable all imperfections (studio-neutral)**
```yaml
breath:        {enabled: false}
vocal_fry:     {enabled: false}
mouth_click:   {enabled: false}
micro_prosody: {enabled: false}
word_life:     {enabled: false}
phrase_swell:  {enabled: false}
```

---

## After changing anything

```bash
python tests/voice_audit.py
```

It renders a corpus and checks clarity, register, naturalness, adaptivity and
text integrity, judging clarity against the raw neural speech so you catch
regressions immediately. Adjust config with your ears **and** the audit.
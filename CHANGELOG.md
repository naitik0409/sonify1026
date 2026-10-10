# CHANGELOG

All notable changes to this project are documented here.

## [Unreleased]

### Integrations
- **MCP server.** `snfy/mcp_server.py` exposes the voice as Model Context
  Protocol tools (`synthesize`, `emotions`) for Claude Desktop, Cursor and any
  MCP client. Install the extra (`pip install "sonify1026[mcp]"`) and run
  `snfy-mcp`. Works with MCP SDK 2.x and 1.x.
- **LangChain & LlamaIndex** tool examples in `examples/`.
- New `docs/INTEGRATIONS.md`; README gains an **Integrations** section.

## [0.1.2] — 2026-10-10

### Packaging
- **`pip install snfy` now works.** A thin alias distribution named `snfy`
  installs the engine (`sonify1026`), so both names install the same code and
  `import snfy` is unchanged.
- GitHub Actions workflows auto-publish `sonify1026` and `snfy` to PyPI when a
  GitHub Release is published (Trusted Publishing or `PYPI_API_TOKEN`).
- Docs: installation and troubleshooting now mention the `snfy` alias.

## [0.1.1] — 2026-10-10

### Voice
- **One voice for every language.** New `voice_mode: single` uses a bilingual
  Indian female speaker (Swara) for English, Hindi and Hinglish, so the
  identity and Indian accent never change at a code-switch. The en-IN voices
  are English-only (they return no audio for Devanagari and drop Hindi words
  in mixed text); the hi-IN voices are genuinely bilingual. A per-script
  `script_rate` trim keeps both languages in the target pace band on one voice.
  `mode: dual` restores the previous behaviour.
- Hindi voice de-thickened for a sweet 20s register: low-mid cuts plus
  presence/air lifts on Swara only, with a +10 Hz voice lift. English untouched.
- Per-voice `rate` offsets pace-match both languages into the same band.
- English articulation quickened (Neerja −7) to lift neutral pace into the
  upper-160s WPM; measured 162–170 while jitter stays calm (med ~4 Hz).

### Realism
- Organic pacing: stronger pause jitter (±28%), more per-segment rate
  drift and frequent fast-burst phrases (0.38, up to +10%) so no two
  sentences share a rhythm.
- **No more shivering.** The 6 Hz amplitude shimmer (`micro_prosody`) and
  phrase-end creak (`vocal_fry`) are now off by default — they read as a
  periodic "shiver", clearest on Hindi/mixed phrase boundaries. Level wander
  and pause jitter keep the delivery alive without the tic.

### Prosody
- Inline emotion tags (`[cheerful]`, `[softly]`, `[warmly]`…) with optional
  per-tag pitch/rate/volume/brightness/warmth overrides; tags never spoken.
- Terminal intonation: gentle fall on declarative ends, lift on questions.
- High-effort content eases slightly instead of rushing.
- Gap compression shortens the backend's over-long internal pauses to
  natural 100–200 ms micro-pauses (click-free).
- Pitch trajectory smoothing caps inter-take steps for seamless code-switches.

### Quality
- Fixed phase-destroying FFT EQ (now `scipy.sosfilt`, phase-preserving).
- Fixed limiter overshoot; ceiling enforced after resampling.
- Removed dead `coarticulation` knob; gated `micro_prosody` call.
- Splice detector now ignores natural plosives (Hindi retroflex stops).
- 20 DSP unit tests + wiring tests; suite now 93 checks.

## [0.1.0] — 2026-10-10

Initial release.

### Voice
- Warm, chest-resonant Indian female register with a pitch guard that verifies
  the rendered voice stays in the intended range.
- Bilingual English + Hindi with automatic Hinglish code-switching.
- Phase-preserving biquad EQ (replaces a phase-destroying FFT implementation).
- No output compression; soft-knee peak limiting enforced after resampling.
- Breath intakes, mouth clicks, vocal fry and micro-pauses, all config-driven.

### Prosody
- Adaptive analysis of intensity words, sentence length and emotional valence.
- Per-sentence modulation of pitch, rate, volume, brightness and warmth.
- Subglottic-effort simulation via content-word emphasis and hedging-word softening.
- Emotional arc across a passage.
- Unique, non-repeating delivery (randomised per call).

### Robustness
- Takes are split only at sentence boundaries, so no word is dropped or repeated.
- Degenerate input (empty, punctuation-only, ellipsis) no longer crashes.
- Hindi digit and number normalisation.

### Testing
- 56 objective pytest checks and a full voice-audit harness covering signal
  integrity, phoneme clarity, register, naturalness, adaptivity and text integrity.
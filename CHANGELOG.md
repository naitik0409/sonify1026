# CHANGELOG

All notable changes to this project are documented here.

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
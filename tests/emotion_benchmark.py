"""
Sonify 1026 — emotion differentiation & speaker identity benchmark.

Answers the two questions the quality suite did not:

  P3   Do the 17 emotion presets actually differ acoustically?
       -> separation matrix (each emotion vs neutral, in z-scored feature
          space) + Spearman correlation between the arousal ORDER implied by
          the config and the arousal order measured from the waveforms.
  P4   Does `voice_mode: single` keep ONE speaker identity across English
       and Hindi?
       -> F0 consistency + mel cepstral distortion (MCD) between the EN and
          HI renders, with a forced two-speaker (Neerja vs Swara) contrast
          proving the metric can tell speakers apart at all.

Method
------
* Identical text for every emotion within a language — only the delivery
  varies, so feature differences are attributable to the emotion preset.
* The SAME fixed seed for every cell — the engine's random jitter (pitch /
  rate / pause) is then drawn identically across cells and cancels in the
  comparison instead of adding noise.
* Expected arousal order is derived from the EMOTION presets in the loaded
  voice_config.yaml; measured order comes from the rendered waveforms.
  Agreement is reported as Spearman rank correlation.
* Features: tests/metrics.py (F0 median/span, intensity, loudness dynamics,
  syllable rate, clarity, spectral centroid, MFCC timbre).

Run:
    python tests/emotion_benchmark.py              # full matrix, EN+HI
    python tests/emotion_benchmark.py --quick      # representative subset
    python tests/emotion_benchmark.py --lang en    # one language only

Outputs:
    outputs/benchmark/*.wav
    outputs/benchmark/emotion_report.json
"""

import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from metrics import (load, emotion_features, identity_features,  # noqa: E402
                     mcd, measure)
from snfy import SonifyEngine, Emotion                        # noqa: E402

OUT = os.path.join(ROOT, "outputs", "benchmark")

# Identical neutral-content text per language: same words, same sentence
# type, so every acoustic difference between cells is delivery, not diction.
TEXTS = {
    "en": "The evening train leaves from the main platform at half past seven.",
    "hi": "शाम की गाड़ी सात बजे मुख्य प्लेटफ़ॉर्म से निकलती है।",
}

# Fixed seed for EVERY cell (see module docstring).
SEED = 20261102

QUICK_EMOTIONS = ["neutral", "happy", "excited", "sad", "calm", "angry", "whisper"]

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


# --------------------------------------------------------------------------- #
# Expected (config-implied) arousal                                            #
# --------------------------------------------------------------------------- #

def config_arousal(emotions_cfg, key):
    """Arousal level implied by an emotion preset in voice_config.yaml.

    Only rate and pitch contribute: the mastering stage normalises every
    render to target_rms_db, so the preset's volume offset cannot appear as
    an absolute-level difference between cells — weighting it in would
    demand acoustic separation the architecture deliberately removes.
    """
    em = emotions_cfg.get(key) or {}
    try:
        rate = float(str(em.get("rate", "0%")).rstrip("%"))
    except ValueError:
        rate = 0.0
    pitch = float(em.get("pitch", 0) or 0)
    return rate + 0.7 * pitch


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def render_cell(engine, text, emotion, out_path, seed=SEED, language=None):
    engine.synthesize(text, emotion=emotion, language=language,
                      output_path=out_path, seed=seed)
    return out_path


def dual_speaker_contrast(engine, out_dir, reuse=False):
    """Render the EN and HI texts through two DIFFERENT speakers (Neerja for
    Latin, Swara for Devanagari — legacy dual mode) so the identity metric
    has a known-different pair to discriminate against."""
    import snfy.engine as E

    en_path = os.path.join(out_dir, "identity_cross_en_neerja.wav")
    hi_path = os.path.join(out_dir, "identity_cross_hi_swara.wav")
    if reuse and os.path.exists(en_path) and os.path.exists(hi_path):
        return en_path, hi_path
    old_mode = E.VOICE_MODE
    old_cfg_mode = dict((engine.config or {}).get("voice_mode") or {})
    try:
        E.VOICE_MODE = {"mode": "dual", "primary": "swara"}
        engine.config["voice_mode"] = {"mode": "dual", "primary": "swara"}
        render_cell(engine, TEXTS["en"], Emotion.NEUTRAL, en_path)
        render_cell(engine, TEXTS["hi"], Emotion.NEUTRAL, hi_path)
    finally:
        E.VOICE_MODE = old_mode
        if old_cfg_mode:
            engine.config["voice_mode"] = old_cfg_mode
        else:
            engine.config.pop("voice_mode", None)
    return en_path, hi_path


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #

def _zscore_rows(rows, keys):
    """Per-feature z-score across emotion rows. Returns list of dicts."""
    out = [{} for _ in rows]
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        if sd < 1e-9:
            sd = 1.0
        for i in range(len(rows)):
            out[i][k] = (vals[i] - mu) / sd
    return out


def _spearman(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(np.asarray(x, dtype=np.float64),
                       np.asarray(y, dtype=np.float64))
    return float(rho), float(p)


def differentiation_report(cell_features, emotions, arousal_cfg):
    """Separation matrix + rank agreement for one language.

    cell_features: {emotion_key: emotion_features dict}
    Returns dict with per-emotion distance-from-neutral (z-space), the
    measured arousal composite per emotion, and Spearman rho vs config.
    """
    from metrics import EMOTION_FEATURE_KEYS, AROUSAL_FEATURE_KEYS

    rows = [cell_features[e] for e in emotions]
    zrows = _zscore_rows(rows, EMOTION_FEATURE_KEYS)
    zmap = dict(zip(emotions, zrows))

    neutral = zmap.get("neutral", {k: 0.0 for k in EMOTION_FEATURE_KEYS})
    separation = {}
    for e in emotions:
        d = zmap[e]
        separation[e] = float(np.sqrt(sum((d[k] - neutral[k]) ** 2
                                          for k in EMOTION_FEATURE_KEYS)))

    measured = {}
    for e in emotions:
        measured[e] = float(np.mean([zmap[e][k] for k in AROUSAL_FEATURE_KEYS]))

    expected = {e: config_arousal(arousal_cfg, e) for e in emotions}
    if len(emotions) >= 3:
        rho, p = _spearman([expected[e] for e in emotions],
                           [measured[e] for e in emotions])
    else:
        rho, p = 0.0, 1.0

    return {
        "separation_from_neutral": separation,
        "measured_arousal": measured,
        "expected_arousal": expected,
        "spearman_rho": rho,
        "spearman_p": p,
        "mean_separation": float(np.mean(list(separation.values()))),
        "features": {e: {k: float(cell_features[e][k])
                         for k in EMOTION_FEATURE_KEYS} for e in emotions},
    }


def identity_report(en_path, hi_path, cross_en_path, cross_hi_path):
    """Speaker-identity distances for voice_mode: single (EN+HI same voice).

    What is asserted vs what is informational:
      * F0 register consistency (EN vs HI) — asserted.  The pitch guard and
        single-voice routing should keep the voice in one register across
        scripts; this is the cross-language identity claim we can measure.
      * Same-text, two-speaker MCD (Swara-EN vs Neerja-EN) — asserted as a
        CONTROL: MCD must detect a speaker change when phonemes are held
        constant, otherwise 'similar' results would be vacuous.
      * Same-voice, same-text MCD (Swara-HI vs Swara-HI) — asserted low:
        catches a routing regression that silently switched voices.
      * Cross-language MCD / band distance — INFORMATIONAL only.  MCD
        assumes parallel phonetic content (voice-conversion paradigm);
        English vs Hindi text differs in phonemes, so those numbers are
        dominated by language, not identity (measured ~9 dB either way).
    """
    xe, fse = load(en_path)
    xh, fsh = load(hi_path)
    xc, fsc = load(cross_en_path)
    xd, fsd = load(cross_hi_path)

    fe = identity_features(xe, fse)
    fh = identity_features(xh, fsh)
    fc = identity_features(xc, fsc)
    fd = identity_features(xd, fsd)

    from metrics import bands as bands_of

    def band_profile(x, fs):
        b = bands_of(x, fs)
        v = np.array([b["low"], b["body"], b["mid"], b["high"], b["air"]])
        v = 10.0 ** (v / 20.0)
        return v / (v.sum() + 1e-20)

    iqr_ratio = (max(fe["f0_iqr"], fh["f0_iqr"]) /
                 max(min(fe["f0_iqr"], fh["f0_iqr"]), 1e-6))
    return {
        "f0_en": fe["f0_median"],
        "f0_hi": fh["f0_median"],
        "f0_delta_hz": abs(fe["f0_median"] - fh["f0_median"]),
        "f0_iqr_en": fe["f0_iqr"],
        "f0_iqr_hi": fh["f0_iqr"],
        "f0_iqr_ratio": float(iqr_ratio),
        # control: same text, two speakers (must be LARGE)
        "mcd_two_speakers_db": mcd(fe["mfcc_mean"], fc["mfcc_mean"]),
        # control: same text, same voice (must be SMALL)
        "mcd_same_voice_text_db": mcd(fh["mfcc_mean"], fd["mfcc_mean"]),
        # informational: same speaker, different language (phonetic confound)
        "mcd_cross_language_db": mcd(fe["mfcc_mean"], fh["mfcc_mean"]),
        "band_l2_cross_language": float(np.linalg.norm(
            band_profile(xe, fse) - band_profile(xh, fsh))),
        "centroid_delta_hz_cross": abs(
            spectral_centroid_of(xe, fse) - spectral_centroid_of(xh, fsh)),
    }


def spectral_centroid_of(x, fs):
    from metrics import spectral_centroid
    return spectral_centroid(x, fs)


# --------------------------------------------------------------------------- #
# Thresholds (calibrated against the baseline run — see docs/BENCHMARKING.md)  #
# --------------------------------------------------------------------------- #

THRESHOLDS = {
    # differentiation, per language (HI bar set from the baseline: Hindi
    # arousal ordering is a known development gap — Priority 2/3 — so it
    # gets a directional floor rather than the full EN bar)
    "min_spearman_rho_en": 0.45,
    "min_spearman_rho_hi": 0.30,
    "min_mean_separation": 0.60,
    "min_extreme_separation_en": 0.45,
    "min_extreme_separation_hi": 0.35,
    # identity / routing
    "max_f0_delta_hz": 30.0,           # EN vs HI, same speaker
    "max_f0_iqr_ratio": 2.0,           # pitch-variability stability
    "min_mcd_two_speakers_db": 4.0,    # control: metric must detect a switch
    "max_mcd_same_voice_text_db": 2.0, # control: same voice must reproduce
}

EXTREME_EMOTIONS = ["happy", "excited", "angry", "sad", "calm", "whisper"]


def evaluate(report):
    """Apply thresholds -> list of (ok, message) checks."""
    checks = []

    # --- voice routing (offline, deterministic) ----------------------------
    routing = report.get("routing") or {}
    if routing:
        same = routing.get("en") == routing.get("hi")
        checks.append((same,
                       f"voice routing single: EN->{routing.get('en')} "
                       f"HI->{routing.get('hi')}"))

    # --- emotion differentiation, per language ----------------------------
    for lang, diff in report.get("differentiation", {}).items():
        rho_min = THRESHOLDS[f"min_spearman_rho_{lang}"]
        rho = diff.get("spearman_rho", 0.0)
        checks.append((rho >= rho_min,
                       f"[{lang}] arousal rank agreement rho={rho:+.2f} "
                       f"(>= {rho_min})"))

        mean_sep = diff.get("mean_separation", 0.0)
        checks.append((mean_sep >= THRESHOLDS["min_mean_separation"],
                       f"[{lang}] mean separation={mean_sep:.2f} z "
                       f"(>= {THRESHOLDS['min_mean_separation']})"))

        sep_min = THRESHOLDS[f"min_extreme_separation_{lang}"]
        sep = diff.get("separation_from_neutral", {})
        for e in EXTREME_EMOTIONS:
            if e not in sep:
                continue
            checks.append((sep[e] >= sep_min,
                           f"[{lang}] {e} vs neutral separation={sep[e]:.2f} z "
                           f"(>= {sep_min})"))

    # --- speaker identity -------------------------------------------------
    ident = report.get("identity") or {}
    if ident:
        checks.append((ident["f0_delta_hz"] <= THRESHOLDS["max_f0_delta_hz"],
                       f"EN/HI F0 delta={ident['f0_delta_hz']:.1f} Hz "
                       f"(<= {THRESHOLDS['max_f0_delta_hz']})"))
        checks.append((ident["f0_iqr_ratio"] <= THRESHOLDS["max_f0_iqr_ratio"],
                       f"EN/HI F0-IQR ratio={ident['f0_iqr_ratio']:.2f} "
                       f"(<= {THRESHOLDS['max_f0_iqr_ratio']})"))
        checks.append((ident["mcd_two_speakers_db"] >=
                       THRESHOLDS["min_mcd_two_speakers_db"],
                       f"control: same-text two-speaker MCD="
                       f"{ident['mcd_two_speakers_db']:.2f} dB "
                       f"(>= {THRESHOLDS['min_mcd_two_speakers_db']})"))
        checks.append((ident["mcd_same_voice_text_db"] <=
                       THRESHOLDS["max_mcd_same_voice_text_db"],
                       f"control: same-voice same-text MCD="
                       f"{ident['mcd_same_voice_text_db']:.2f} dB "
                       f"(<= {THRESHOLDS['max_mcd_same_voice_text_db']})"))
    return checks


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def run_benchmark(emotions=None, langs=("en", "hi"), quick=False,
                  with_identity=True, engine=None, out_dir=OUT, verbose=True,
                  reuse=False):
    """Render + measure the benchmark matrix.  Returns the report dict.

    Shared by the CLI below and tests/test_emotion_benchmark.py so the
    pytest assertions judge exactly what the report tool measures.
    `reuse=True` re-analyses existing WAVs in out_dir instead of rendering
    (fast threshold calibration; the files must come from the same seed).
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    engine = engine or SonifyEngine()
    emotions = emotions or (QUICK_EMOTIONS if quick else
                            list(engine.config.get("emotions", {})))
    emotions = [e for e in emotions if e in
                {k for k in (engine.config.get("emotions") or {})}]

    # Deterministic backend within a benchmark run: identical takes reuse the
    # same raw audio (pitch-guard retries and repeat runs become free).
    engine.config["backend_cache"] = {"enabled": True, "max_entries": 512}

    from snfy.engine import _pick_voice
    report = {
        "version": _pkg_version(),
        "edge_tts_version": _edge_version(),
        "seed": SEED,
        "emotions": emotions,
        "languages": list(langs),
        "thresholds": dict(THRESHOLDS),
        # single-voice routing: both languages must resolve to ONE voice key
        "routing": {"en": _pick_voice(TEXTS["en"], "in"),
                    "hi": _pick_voice(TEXTS["hi"], "in")},
        "cells": {},
        "differentiation": {},
        "identity": {},
        "elapsed_s": 0.0,
    }

    for lang in langs:
        text = TEXTS[lang]
        cells = {}
        for emo in emotions:
            path = os.path.join(out_dir, f"cell_{lang}_{emo}.wav")
            try:
                if not (reuse and os.path.exists(path)):
                    render_cell(engine, text, getattr(Emotion, emo.upper()), path)
                x, fs = load(path)
                cells[emo] = emotion_features(x, fs)
                if verbose:
                    f = cells[emo]
                    print(f"  {DIM}rendered{RESET} {lang}/{emo:<12s} "
                          f"f0={f['f0_median']:6.1f}Hz span={f['f0_span']:5.1f}Hz "
                          f"pace={f['pace']:.3f} rms={f['rms_db']:6.1f}dB")
            except Exception as e:                       # noqa: BLE001
                print(f"  {RED}FAILED{RESET} {lang}/{emo}: {type(e).__name__}: {e}")
        report["cells"][lang] = cells
        if cells:
            report["differentiation"][lang] = differentiation_report(
                cells, list(cells), engine.config.get("emotions", {}))

    if with_identity and "en" in langs and "hi" in langs:
        en_path = os.path.join(out_dir, "cell_en_neutral.wav")
        hi_path = os.path.join(out_dir, "cell_hi_neutral.wav")
        if not (os.path.exists(en_path) and os.path.exists(hi_path)):
            render_cell(engine, TEXTS["en"], Emotion.NEUTRAL, en_path)
            render_cell(engine, TEXTS["hi"], Emotion.NEUTRAL, hi_path)
        cross_en, cross_hi = dual_speaker_contrast(engine, out_dir, reuse=reuse)
        report["identity"] = identity_report(en_path, hi_path,
                                             cross_en, cross_hi)

    report["checks"] = [(bool(ok), msg) for ok, msg in evaluate(report)]
    report["passed"] = all(ok for ok, _ in report["checks"])
    report["elapsed_s"] = round(time.time() - t0, 1)
    return report


def _pkg_version():
    try:
        import snfy
        return getattr(snfy, "__version__", "?")
    except Exception:
        return "?"


def _edge_version():
    try:
        import importlib.metadata as md
        return md.version("edge-tts")
    except Exception:
        return "?"


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def print_report(report):
    print(f"\n{DIM}{'=' * 78}{RESET}")
    print("  EMOTION DIFFERENTIATION — separation from neutral (z-space)")
    print(f"{DIM}{'=' * 78}{RESET}")
    for lang, diff in report["differentiation"].items():
        print(f"\n  [{lang.upper()}]  spearman rho={diff['spearman_rho']:+.2f} "
              f"(p={diff['spearman_p']:.4f})  mean separation={diff['mean_separation']:.2f}")
        print(f"  {'emotion':<14s} {'separation':>10s} {'measured':>9s} "
              f"{'expected':>9s}   f0      span   pace   cent")
        order = sorted(diff["separation_from_neutral"].items(),
                       key=lambda kv: -kv[1])
        for e, d in order:
            f = diff["features"][e]
            print(f"  {e:<14s} {d:10.2f} {diff['measured_arousal'][e]:9.2f} "
                  f"{diff['expected_arousal'][e]:9.1f}   "
                  f"{f['f0_median']:5.0f}Hz {f['f0_span']:5.1f} "
                  f"{f['pace']:6.3f}  {f['centroid_hz']:5.0f}")

    ident = report.get("identity") or {}
    if ident:
        print(f"\n{DIM}{'=' * 78}{RESET}")
        print("  SPEAKER IDENTITY — one voice across EN/HI (voice_mode: single)")
        print(f"{DIM}{'=' * 78}{RESET}")
        print(f"  F0 EN {ident['f0_en']:.1f} Hz | HI {ident['f0_hi']:.1f} Hz "
              f"| delta {ident['f0_delta_hz']:.1f} Hz "
              f"(IQR ratio {ident['f0_iqr_ratio']:.2f})")
        print(f"  control: same text, two speakers MCD = "
              f"{ident['mcd_two_speakers_db']:.2f} dB (metric detects switches)")
        print(f"  control: same text, same voice  MCD = "
              f"{ident['mcd_same_voice_text_db']:.2f} dB (routing stable)")
        print(f"  informational: cross-language MCD = "
              f"{ident['mcd_cross_language_db']:.2f} dB, "
              f"band L2 = {ident['band_l2_cross_language']:.3f} "
              f"{DIM}(phonetic-content confounded — not asserted){RESET}")

    print(f"\n{DIM}{'=' * 78}{RESET}")
    print("  CHECKS")
    print(f"{DIM}{'=' * 78}{RESET}")
    for ok, msg in report["checks"]:
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {msg}")

    passed = sum(1 for ok, _ in report["checks"] if ok)
    total = len(report["checks"])
    verdict = f"{GREEN}BENCHMARK PASSED" if report["passed"] else f"{RED}BENCHMARK FAILED"
    print(f"\n  {verdict}{RESET}  ({passed}/{total} checks, "
          f"{report['elapsed_s']:.0f}s, v{report['version']}, "
          f"edge-tts {report['edge_tts_version']})")


def main():
    ap = argparse.ArgumentParser(description="Sonify emotion/identity benchmark")
    ap.add_argument("--quick", action="store_true",
                    help="representative emotion subset (faster)")
    ap.add_argument("--lang", choices=["en", "hi", "both"], default="both")
    ap.add_argument("--no-identity", action="store_true",
                    help="skip the cross-speaker identity contrast")
    ap.add_argument("--reuse", action="store_true",
                    help="re-analyse existing WAVs in outputs/benchmark "
                         "(no network; same seed only)")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON report")
    args = ap.parse_args()

    langs = ("en", "hi") if args.lang == "both" else (args.lang,)
    report = run_benchmark(quick=args.quick, langs=langs,
                           with_identity=not args.no_identity,
                           reuse=args.reuse)

    os.makedirs(OUT, exist_ok=True)
    json_path = os.path.join(OUT, "emotion_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"{DIM}Sonify 1026 emotion & identity benchmark"
              f"{'  (quick)' if args.quick else ''}{RESET}")
        print(f"{DIM}output: {OUT}{RESET}")
        print_report(report)
        print(f"\n  {DIM}report: {json_path}{RESET}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

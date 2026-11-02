"""
Sonify 1026 — emotion differentiation & speaker identity benchmark tests.

Offline section (no network): unit-tests the benchmark metrics themselves on
synthetic signals where ground truth is known, plus the pure report logic.

Live section (network): renders the benchmark matrix through Edge TTS and
asserts the thresholds in tests/emotion_benchmark.py — this is the objective
answer to "do the 17 emotions actually differ?" (P3) and "is it one speaker
across English and Hindi?" (P4).

Run:
    pytest tests/test_emotion_benchmark.py -q -m network   # live part
    pytest tests/test_emotion_benchmark.py -q               # offline part
      (live tests are skipped automatically without -m network? No — they
       are marked, run with `-m network` to include, `-m "not network"` to
       exclude; a plain run executes everything and needs the network.)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import (mfcc, mcd, cepstral_mean, emotion_features,  # noqa: E402
                     identity_features, syllable_nuclei, load)
from emotion_benchmark import (THRESHOLDS, differentiation_report,  # noqa: E402
                               config_arousal, run_benchmark, TEXTS)
from snfy import SonifyEngine                                    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_pytest", "benchmark")

# Representative high/low-arousal subset for the live run (the CLI tool
# covers all 17; pytest keeps the matrix to a reviewable size).
LIVE_EMOTIONS = ["neutral", "warm", "happy", "excited",
                 "sad", "calm", "angry", "whisper"]


def _tone(freq, fs=48000, dur=0.8, amp=0.3, harmonics=(1, 2, 3, 4)):
    t = np.arange(int(fs * dur)) / fs
    x = sum(amp * np.sin(2 * np.pi * freq * k * t) / (k + 1) for k in harmonics)
    return x.astype(np.float64)


# =========================================================================== #
# 1. Metric units — offline, synthetic ground truth                           #
# =========================================================================== #

class TestBenchmarkMetricUnits:
    def test_mfcc_shape_and_determinism(self):
        x = _tone(220.0)
        c1, c2 = mfcc(x, 48000), mfcc(x, 48000)
        assert c1.shape[1] == 13
        assert c1.shape[0] > 10, "too few frames for a 0.8 s signal"
        assert np.allclose(c1, c2), "MFCC extraction must be deterministic"
        assert np.all(np.isfinite(c1))

    def test_mcd_zero_for_identical_timbre(self):
        x = _tone(220.0)
        c = cepstral_mean(x, 48000)
        assert mcd(c, c) < 1e-9

    def test_mcd_separates_different_timbres(self):
        """Two clearly different sources must score far apart on MCD —
        otherwise the identity metric cannot discriminate speakers."""
        a = cepstral_mean(_tone(180.0, harmonics=(1, 2, 3)), 48000)
        b = cepstral_mean(_tone(320.0, harmonics=(1, 3, 5)), 48000)
        assert mcd(a, b) > 1.0, "MCD failed to separate distinct timbres"

    def test_emotion_features_f0_matches_synthetic_tone(self):
        fs = 48000
        x = _tone(220.0, fs=fs, dur=1.2)
        f = emotion_features(x, fs)
        assert abs(f["f0_median"] - 220.0) < 12.0, \
            f"F0 {f['f0_median']:.1f} should track the 220 Hz tone"
        assert np.isfinite(f["rms_db"]) and np.isfinite(f["centroid_hz"])
        assert f["duration"] == pytest.approx(1.2, abs=0.05)

    def test_syllable_nuclei_counts_bursts(self):
        fs = 48000
        # 6 well-separated energy bursts -> 6 nuclei
        pieces, gap = [], np.zeros(int(fs * 0.30))
        for _ in range(6):
            pieces.append(_tone(200.0, fs=fs, dur=0.15, amp=0.5))
            pieces.append(gap)
        x = np.concatenate(pieces)
        n = syllable_nuclei(x, fs)
        assert n == 6, f"expected 6 syllable nuclei, counted {n}"

    def test_identity_features_include_timbre_vector(self):
        f = identity_features(_tone(220.0), 48000)
        assert f["mfcc_mean"].shape == (13,)
        assert f["f0_median"] > 0


# =========================================================================== #
# 2. Report logic — offline, synthetic rows                                   #
# =========================================================================== #

class TestBenchmarkReportLogic:
    def test_config_arousal_ordering_matches_intent(self):
        """The shipped config must order arousal as the voice brief intends:
        excited/angry high, sad/calm/whisper low, neutral in between."""
        import snfy.engine as E
        emos = E.EMOTIONS
        a = {k: config_arousal(emos, k)
             for k in ("excited", "angry", "neutral", "calm", "sad", "whisper")}
        assert a["excited"] > a["neutral"] > a["sad"], a
        assert a["angry"] > a["neutral"], a
        assert a["whisper"] < a["neutral"], a
        assert a["calm"] < a["neutral"], a

    def test_differentiation_report_perfect_monotone(self):
        """Featuring rows built to follow the expected arousal order must
        yield rho ~ 1 and non-trivial separation from neutral.

        The row order is taken from the config itself so the test measures
        the report machinery, not a hand-guessed emotion ranking."""
        keys = ["f0_median", "f0_span", "rms_db", "dyn_db",
                "rate_syll", "clarity", "centroid_hz"]
        import snfy.engine as E
        order = sorted(["sad", "calm", "whisper", "neutral", "happy", "excited"],
                       key=lambda k: config_arousal(E.EMOTIONS, k))
        rows = {}
        for i, e in enumerate(order):
            base = {k: 0.0 for k in keys}
            base["f0_median"] = 180.0 + 15.0 * i
            base["f0_span"] = 20.0 + 8.0 * i
            base["rms_db"] = -28.0 + 2.0 * i
            base["dyn_db"] = 3.0 + 1.5 * i
            base["rate_syll"] = 2.0 + 0.4 * i
            base["clarity"] = 6.0 + 0.3 * i
            base["centroid_hz"] = 2500.0 + 200.0 * i
            rows[e] = base
        rep = differentiation_report(rows, order, E.EMOTIONS)
        assert rep["spearman_rho"] > 0.95, rep["spearman_rho"]
        assert rep["separation_from_neutral"]["excited"] > \
            rep["separation_from_neutral"].get("neutral", 0.0)
        assert rep["mean_separation"] > 0.5

    def test_thresholds_are_sane(self):
        assert THRESHOLDS["min_spearman_rho"] > 0
        assert THRESHOLDS["max_f0_delta_hz"] < 60
        assert THRESHOLDS["max_mcd_ratio"] < 1.0


# =========================================================================== #
# 3. Live benchmark — network required (Edge TTS)                             #
# =========================================================================== #

@pytest.mark.network
class TestEmotionDifferentiationLive:
    @pytest.fixture(scope="class")
    def report(self):
        os.makedirs(OUT, exist_ok=True)
        return run_benchmark(emotions=LIVE_EMOTIONS, langs=("en", "hi"),
                             with_identity=True, out_dir=OUT, verbose=False)

    def test_all_cells_rendered(self, report):
        for lang in ("en", "hi"):
            cells = report["cells"].get(lang, {})
            missing = [e for e in LIVE_EMOTIONS if e not in cells]
            assert not missing, f"{lang}: failed to render {missing}"
            for e, f in cells.items():
                assert f["f0_median"] > 100.0, \
                    f"{lang}/{e}: no voiced pitch (f0={f['f0_median']})"
                assert f["duration"] > 0.3, f"{lang}/{e}: too short"

    def test_emotions_differ_acoustically_english(self, report):
        diff = report["differentiation"]["en"]
        assert diff["spearman_rho"] >= THRESHOLDS["min_spearman_rho"], \
            f"config-vs-measured arousal rho={diff['spearman_rho']:+.2f} " \
            f"— emotion presets are not ordering the output as configured"
        assert diff["mean_separation"] >= THRESHOLDS["min_mean_separation"], \
            f"mean separation {diff['mean_separation']:.2f} z — renders are " \
            f"acoustically interchangeable across emotions"
        for e in ("happy", "excited", "sad", "calm"):
            s = diff["separation_from_neutral"].get(e, 0.0)
            assert s >= THRESHOLDS["min_extreme_separation"], \
                f"{e} is only {s:.2f} z from neutral (indistinguishable)"

    def test_emotions_differ_acoustically_hindi(self, report):
        diff = report["differentiation"]["hi"]
        assert diff["spearman_rho"] >= THRESHOLDS["min_spearman_rho"], \
            f"HI rho={diff['spearman_rho']:+.2f}"
        assert diff["mean_separation"] >= THRESHOLDS["min_mean_separation"], \
            f"HI mean separation {diff['mean_separation']:.2f} z"
        for e in ("happy", "sad", "whisper"):
            s = diff["separation_from_neutral"].get(e, 0.0)
            assert s >= THRESHOLDS["min_extreme_separation"], \
                f"HI {e} is only {s:.2f} z from neutral"

    def test_single_speaker_identity_across_languages(self, report):
        ident = report["identity"]
        assert ident, "identity section missing from report"
        assert ident["f0_delta_hz"] <= THRESHOLDS["max_f0_delta_hz"], \
            f"EN/HI F0 differs by {ident['f0_delta_hz']:.1f} Hz — " \
            f"voice_mode:single is not holding one register across scripts"
        assert ident["mcd_same_speaker_db"] <= THRESHOLDS["max_mcd_same_db"], \
            f"same-speaker MCD {ident['mcd_same_speaker_db']:.2f} dB — " \
            f"timbre shifts too far between English and Hindi"

    def test_identity_metric_discriminates_speakers(self, report):
        """Control: the MCD metric must rate the forced two-speaker pair as
        MORE different than the same-speaker EN/HI pair — otherwise the
        'same speaker' result above would be vacuous."""
        ident = report["identity"]
        assert ident["mcd_ratio"] <= THRESHOLDS["max_mcd_ratio"], \
            f"same/two-speaker MCD ratio {ident['mcd_ratio']:.2f}: metric " \
            f"cannot distinguish one speaker from two"
        assert ident["mcd_two_speaker_db"] > ident["mcd_same_speaker_db"]

    def test_report_persisted(self, report):
        assert len(report["checks"]) >= 8, "threshold checks missing from report"
        assert report["elapsed_s"] > 0
        assert report["edge_tts_version"]

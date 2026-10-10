"""Deterministic DSP unit tests + config-wiring tests.

Every humanization knob is verified two ways (the backend TTS service is
non-deterministic, so output comparison can never prove anything):
  1. the DSP function itself behaves correctly on synthetic input, and
  2. the render path actually invokes it when enabled, and skips it when
     disabled (verified by call counting, not by audio comparison).
"""
import copy

import numpy as np
import pytest

import snfy.engine as E
from snfy import SonifyEngine, Emotion

FS = 48000


def tone(seconds=1.0, freq=220.0, amp=0.4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * seconds)) / FS
    x = amp * np.sin(2 * np.pi * freq * t)
    x += 0.01 * rng.standard_normal(len(t))
    return x.astype(np.float64)


def words_for(x, fs=FS):
    # two fake word spans covering the first and middle thirds
    n = len(x)
    return [(0.05, 0.25, "absolutely"), (n / fs * 0.4, 0.25, "wonderful")]


# --------------------------------------------------------------------------- #
# 1. Unit behaviour on synthetic input                                          #
# --------------------------------------------------------------------------- #

class TestDspUnits:
    def test_breath_shape_and_level(self):
        rng = np.random.default_rng(1)
        cfg = {"duration_min": 0.14, "duration_max": 0.24,
               "level_normal": 0.11, "level_emotional": 0.14}
        b = E._breath(FS, rng, ref_rms=0.1, emotion="neutral", config=cfg)
        assert b is not None and len(b) > FS * 0.10
        # fades at both edges (no click when spliced)
        edge = max(1, int(FS * 0.006))
        assert np.max(np.abs(b[:edge])) < np.max(np.abs(b)) * 0.5
        assert np.max(np.abs(b[-edge:])) < np.max(np.abs(b)) * 0.5
        # level tracks ref_rms
        b2 = E._breath(FS, np.random.default_rng(1), ref_rms=0.2,
                       emotion="neutral", config=cfg)
        assert abs(np.sqrt(np.mean(b2 ** 2)) / np.sqrt(np.mean(b ** 2)) - 2.0) < 0.3

    def test_breath_exhale_variant(self):
        rng = np.random.default_rng(2)
        b = E._breath(FS, rng, ref_rms=0.1, emotion="sad", config={}, exhale=True)
        assert b is not None and np.all(np.isfinite(b))

    def test_mouth_click_is_a_short_transient(self):
        rng = np.random.default_rng(3)
        c = E._mouth_click(FS, rng, duration_ms=6.0)
        assert 0 < len(c) <= int(FS * 0.02)
        assert np.all(np.isfinite(c))
        # decays: second half quieter than first
        assert np.sqrt(np.mean(c[len(c) // 2:] ** 2)) < np.sqrt(np.mean(c[:len(c) // 2] ** 2))

    def test_vocal_fry_touches_only_the_tail(self):
        x = tone(1.0)
        rng = np.random.default_rng(4)
        cfg = {"enabled": True, "intensity": 0.18, "duration_frac": 0.16,
               "max_seconds": 0.40, "pulse_hz": 34.0}
        y = E._apply_vocal_fry(x, FS, rng, cfg, strength=1.0)
        assert len(y) == len(x)
        head = int(len(x) * 0.7)
        assert np.allclose(y[:head], x[:head]), "fry must not touch the head"
        assert not np.allclose(y[head:], x[head:]), "fry must shape the tail"

    def test_word_life_needs_words(self):
        x = tone(1.0)
        rng = np.random.default_rng(5)
        assert np.array_equal(E._word_life(x, FS, [], rng, db=1.4), x)
        y = E._word_life(x, FS, words_for(x), np.random.default_rng(5), db=1.4)
        assert not np.allclose(y, x), "word_life must modulate with word times"

    def test_phrase_swell_modulates(self):
        x = tone(1.0)
        y = E._phrase_swell(x, FS, np.random.default_rng(6), db=1.6)
        assert len(y) == len(x) and not np.allclose(y, x)

    def test_micro_prosody_respects_enabled(self):
        x = tone(1.0)
        assert np.array_equal(
            E._apply_micro_prosody(x, FS, np.random.default_rng(7), {"enabled": False}), x)
        y = E._apply_micro_prosody(
            x, FS, np.random.default_rng(7),
            {"enabled": True, "amplitude_shimmer_db": 0.6, "rate_hz": 6.0})
        assert not np.allclose(y, x)

    def test_keyword_emphasis_lifts_key_words_only(self):
        x = tone(1.0)
        words = [(0.05, 0.20, "absolutely"), (0.60, 0.20, "plain")]
        rng = np.random.default_rng(8)
        y = E._keyword_emphasis(x, FS, words, rng, ["absolutely"],
                                {"boost_db": 2.0}, low_words=[])
        a0, b0 = int(0.05 * FS), int(0.25 * FS)
        a1, b1 = int(0.60 * FS), int(0.80 * FS)
        assert np.sqrt(np.mean(y[a0:b0] ** 2)) > np.sqrt(np.mean(x[a0:b0] ** 2)) * 1.05
        assert np.allclose(y[a1:b1], x[a1:b1], atol=1e-9), "non-key region must be untouched"

    def test_keyword_softening(self):
        x = tone(1.0)
        words = [(0.05, 0.20, "maybe")]
        y = E._keyword_emphasis(x, FS, words, np.random.default_rng(9), [],
                                {"boost_db": 2.0, "soften_db": -1.2},
                                low_words=["maybe"])
        a0, b0 = int(0.05 * FS), int(0.25 * FS)
        assert np.sqrt(np.mean(y[a0:b0] ** 2)) < np.sqrt(np.mean(x[a0:b0] ** 2))

    def test_breath_before_keyword_inserts_and_preserves(self):
        x = tone(1.0)
        words = [(0.50, 0.20, "absolutely")]
        bcfg = {"duration_min": 0.14, "duration_max": 0.24,
                "level_normal": 0.11, "level_emotional": 0.14}
        y = E._breath_before_keyword(x, FS, words, np.random.default_rng(10),
                                     0.1, "neutral", bcfg, ["absolutely"])
        assert len(y) > len(x), "breath must extend the signal"
        # content after the insertion point is preserved (shifted, not smeared)
        tail_src = x[int(0.70 * FS):int(0.90 * FS)]
        best = 0.0
        for off in range(0, len(y) - len(tail_src), FS // 100):
            seg = y[off:off + len(tail_src)]
            c = abs(float(np.dot(seg, tail_src)) /
                    (np.linalg.norm(seg) * np.linalg.norm(tail_src) + 1e-12))
            best = max(best, c)
        assert best > 0.90, "original content must survive the insertion"

    def test_emphasis_markers_boost_matched_words(self):
        x = tone(1.0)
        words = [(0.05, 0.20, "absolutely"), (0.60, 0.20, "plain")]
        y = E._emphasis_env(x, FS, words, ["absolutely"], boost_db=2.5, ramp_factor=0.35)
        a0, b0 = int(0.05 * FS), int(0.25 * FS)
        a1, b1 = int(0.60 * FS), int(0.80 * FS)
        assert np.sqrt(np.mean(y[a0:b0] ** 2)) > np.sqrt(np.mean(x[a0:b0] ** 2)) * 1.1
        assert np.allclose(y[a1:b1], x[a1:b1], atol=1e-9)
        # no match => identity
        assert np.array_equal(E._emphasis_env(x, FS, words, ["zzz"], boost_db=2.5), x)

    def test_estimate_f0_agrees_with_metrics(self):
        import os
        import soundfile as sf
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from metrics import pitch_stats
        x = tone(2.0, freq=220.0)
        est = E._estimate_f0(x, FS, sample_seconds=2.0)
        assert 200.0 < est < 240.0, "F0 estimate %.1f Hz off a 220 Hz tone" % est

    def test_pitch_arc_zero_contour_is_identity(self):        # Proves the unconditional render-path call is safe when disabled.
        x = tone(1.0)
        y = E._apply_pitch_arc(x, FS, np.zeros(140))
        assert np.allclose(y, x, atol=1e-9)

    def test_compress_gaps_exact_accounting(self):
        t = np.arange(FS * 2) / FS
        speech = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)
        x = np.concatenate([speech, np.zeros(int(FS * 1.2)), speech,
                            np.zeros(int(FS * 0.1)), speech])
        cfg = {"gap_compress": {"enabled": True, "threshold_db": -50.0,
                                "max_silence_ms": 220.0, "target_ms": 160.0,
                                "fade_ms": 12.0}}
        y = E._compress_gaps(x, FS, cfg)
        assert abs(len(y) / FS - 6.248) < 0.02
        d = np.max(np.abs(np.diff(y)))
        assert d < 0.15, "crossfade must not click (jump %.4f)" % d
        assert np.allclose(y[:FS * 2], speech[:FS * 2])
        assert np.allclose(y[-FS * 2:], speech[-FS * 2:])

    def test_compress_gaps_disabled_is_identity(self):
        x = tone(0.5)
        assert np.array_equal(
            E._compress_gaps(x, FS, {"gap_compress": {"enabled": False}}), x)

    def test_apply_terminal_shapes_final_only(self):
        cfg = {"sentence_type_rules": {"sentence": {"terminal_pitch": -2},
                                       "question": {"terminal_pitch": 2}},
               "character": {}}
        vkey = {"vkey": "neerja", "pitch": 0}
        E.VOICES.setdefault("neerja", {"pitch": 0, "lo": -45, "hi": 25})
        segs = [dict(vkey="neerja", pitch=0, kind="sentence", text="a."),
                dict(vkey="neerja", pitch=0, kind="sentence", text="b.")]
        E._apply_terminal(segs, cfg)
        assert segs[0]["pitch"] == 0, "non-final take must be untouched"
        assert segs[1]["pitch"] == -2, "final declarative must drop"
        q = [dict(vkey="neerja", pitch=0, kind="question", text="a?")]
        E._apply_terminal(q, cfg)
        assert q[0]["pitch"] == 2, "final question must lift"
        f = [dict(vkey="neerja", pitch=0, kind="sentence", text="x.", is_filler=True)]
        E._apply_terminal(f, cfg)
        assert f[0]["pitch"] == 0, "fillers must be skipped"
        E._apply_terminal([], cfg)  # must not raise

    def test_split_emotion_tags(self):
        cfg_tags = {"cheerful": "happy", "softly": "gentle"}
        spans = E._split_emotion_tags("[cheerful] Hi there. [softly] Come here.", cfg_tags)
        assert spans == [("happy", {}, " Hi there. "), ("gentle", {}, " Come here.")]
        # unknown tags pass through untouched
        spans = E._split_emotion_tags("Hello [foobar] world.", cfg_tags)
        assert spans == [(None, {}, "Hello [foobar] world.")]
        # dict-form overrides survive
        tags2 = {"cheerful": {"emotion": "happy", "pitch_hz": 10, "rate_pct": 5}}
        spans = E._split_emotion_tags("[cheerful] Hi.", tags2)
        assert spans[0][0] == "happy" and spans[0][1] == {"pitch_hz": 10, "rate_pct": 5}


# --------------------------------------------------------------------------- #
# 2. Config wiring: flags must control invocation (call counting, not audio)    #
# --------------------------------------------------------------------------- #

TEXT = ("Good morning everyone. This is absolutely wonderful news for all of us "
        "today, and I really appreciate everything.")


def _count_calls(monkeypatch, engine, mutations):
    """Run one short synthesis, counting DSP invocations. Returns the counts."""
    import copy
    calls = {}
    saved = {}

    def wrap(name):
        orig = getattr(E, name)
        saved[name] = orig

        def fn(*a, **k):
            calls[name] = calls.get(name, 0) + 1
            return orig(*a, **k)
        return fn

    for name in ["_breath", "_mouth_click", "_apply_vocal_fry", "_word_life",
                 "_phrase_swell", "_apply_micro_prosody", "_keyword_emphasis",
                 "_breath_before_keyword"]:
        monkeypatch.setattr(E, name, wrap(name), raising=True)

    base = copy.deepcopy(engine.config)
    try:
        for path, value in mutations:
            node = engine.config
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
        engine._apply_config()
        engine.synthesize(TEXT, emotion=Emotion.WARM,
                          output_path="outputs/_wire.wav", seed=4242)
    finally:
        engine.config = base
        engine._apply_config()
    return calls


@pytest.fixture(scope="module")
def engine():
    import os
    os.makedirs("outputs", exist_ok=True)
    return SonifyEngine()


class TestWiring:
    def test_all_features_fire_when_enabled(self, engine, monkeypatch):
        calls = _count_calls(monkeypatch, engine, [
            (["breath", "sentence_prob"], 1.0),
            (["breath", "question_prob"], 1.0),
            (["breath", "default_prob"], 1.0),
            (["mouth_click", "probability"], 1.0),
            (["vocal_fry", "enabled"], True),
            (["vocal_fry", "probability"], 1.0),
            (["micro_prosody", "enabled"], True),
            (["keyword_breath", "probability"], 1.0),
        ])
        for name in ["_breath", "_mouth_click", "_apply_vocal_fry", "_word_life",
                     "_phrase_swell", "_apply_micro_prosody", "_keyword_emphasis",
                     "_breath_before_keyword"]:
            assert calls.get(name, 0) >= 1, f"{name} never invoked with feature on"

    def test_all_features_silent_when_disabled(self, engine, monkeypatch):
        calls = _count_calls(monkeypatch, engine, [
            (["breath", "enabled"], False),
            (["mouth_click", "enabled"], False),
            (["vocal_fry", "enabled"], False),
            (["word_life", "enabled"], False),
            (["phrase_swell", "enabled"], False),
            (["micro_prosody", "enabled"], False),
            (["keyword_emphasis", "enabled"], False),
            (["keyword_breath", "enabled"], False),
        ])
        for name in ["_breath", "_mouth_click", "_apply_vocal_fry", "_word_life",
                     "_phrase_swell", "_apply_micro_prosody", "_keyword_emphasis",
                     "_breath_before_keyword"]:
            assert calls.get(name, 0) == 0, f"{name} invoked with feature off"

    def test_pitch_guard_rerenders_when_out_of_range(self, engine, monkeypatch):
        import copy
        calls = []

        orig_render = E.render

        def spy(*a, **k):
            calls.append(1)
            return orig_render(*a, **k)

        monkeypatch.setattr(E, "render", spy, raising=True)
        base = copy.deepcopy(engine.config)
        try:
            # absurd floor forces at least one correction re-render
            engine.config["pitch_guard"]["min_f0_hz"] = 400.0
            engine.config["pitch_guard"]["max_attempts"] = 2
            engine._apply_config()
            engine.synthesize("Hello there, this is a short test.",
                              emotion=Emotion.NEUTRAL,
                              output_path="outputs/_guard.wav", seed=999)
            assert len(calls) >= 2, "guard did not re-render out-of-range audio"
        finally:
            engine.config = base
            engine._apply_config()


# --------------------------------------------------------------------------- #
# 3. Single-voice mode: one identity for every language                        #
# --------------------------------------------------------------------------- #

class TestSingleVoice:
    def test_single_mode_uses_primary_for_every_script(self):
        saved = dict(E.VOICE_MODE)
        try:
            E.VOICE_MODE = {"mode": "single", "primary": "swara"}
            en = E._pick_voice("Good morning, my friend.", "in")
            hi = E._pick_voice("नमस्ते, आप कैसे हैं?", "in")
            hing = E._pick_voice("Yaar kal meeting hai.", "in")
            mix = E._pick_voice("Namaste, आज का meeting है.", "in")
            assert en == hi == hing == mix == "swara", \
                "single mode must return one voice for all scripts"
        finally:
            E.VOICE_MODE = saved

    def test_dual_mode_splits_by_script(self):
        saved = dict(E.VOICE_MODE)
        try:
            E.VOICE_MODE = {"mode": "dual", "primary": "swara"}
            en = E._pick_voice("Good morning, my friend.", "in")
            hi = E._pick_voice("नमस्ते, आप कैसे हैं?", "in")
            assert en == "neerja" and hi == "swara", \
                "dual mode must keep the legacy English/Devanagari split"
        finally:
            E.VOICE_MODE = saved

    def test_default_config_is_single_voice(self, engine):
        assert engine.config["voice_mode"]["mode"] == "single"
        primary = engine.config["voice_mode"]["primary"]
        assert primary in engine.config["voices"], \
            "voice_mode.primary must name a configured voice"
        assert "script_rate" in engine.config["voice_mode"]

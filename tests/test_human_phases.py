"""Tests for the fully-human voice phases A-G.

The backend TTS service is non-deterministic, so each phase is verified at
the DSP/analysis level on synthetic input (deterministic), plus one render
integration test that exercises the wiring end to end.
"""
import copy

import numpy as np

import snfy
import snfy.engine as E

FS = 48000


def noise(seconds=1.0, amp=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(FS * seconds))).astype(np.float64)


def rms(z):
    return float(np.sqrt(np.mean(z ** 2) + 1e-12))


def hf_energy(z, lo=0.5, hi=0.7, cutoff=3000.0):
    seg = z[int(lo * FS):int(hi * FS)]
    spec = np.abs(np.fft.rfft(seg))
    freq = np.fft.rfftfreq(len(seg), 1.0 / FS)
    return float(spec[freq > cutoff].sum())


# --------------------------------------------------------------------------- #
# Phase A — clause-level takes                                                  #
# --------------------------------------------------------------------------- #

class TestClauseTakes:
    def test_clause_takes_splits_long_sentence(self):
        text = ("Alpha one, bravo two, charlie three, delta four, "
                "echo five, foxtrot six, golf seven.")
        off = {"humanize": {"max_utterance_chars": 40}}
        on = {"humanize": {
            "max_utterance_chars": 40, "soft_max_utterance_chars": 520,
            "clause_takes": {"enabled": True, "max_utterance_chars": 40}}}
        base = E._pack_utterances(text, max_chars=40, config=off)
        clause = E._pack_utterances(text, max_chars=40, config=on)
        assert len(clause) > len(base)

    def test_clause_takes_are_bounded(self):
        text = "One two, three four, five six, seven eight, nine ten, eleven twelve."
        on = {"humanize": {
            "max_utterance_chars": 20, "soft_max_utterance_chars": 520,
            "clause_takes": {"enabled": True, "max_utterance_chars": 20}}}
        takes = E._pack_utterances(text, max_chars=20, config=on)
        assert all(t for t, _ in takes)
        # no take should wildly exceed the cap (a single clause may, but not all)
        assert sum(len(t) <= 40 for t, _ in takes) >= len(takes) - 1


# --------------------------------------------------------------------------- #
# Phase B — given-information de-accenting                                      #
# --------------------------------------------------------------------------- #

class TestDeAccent:
    def test_given_words_are_softened(self):
        x = noise(1.0, seed=1)
        wt = [(0.5, 0.2, "plan")]
        cfg = {"enabled": True, "given_soften_db": -3.0}
        y = E._de_accent(x, FS, wt, np.random.default_rng(0), ["plan"], cfg)
        a, b = int(0.5 * FS), int(0.7 * FS)
        assert rms(y[a:b]) < rms(x[a:b]) * 0.9

    def test_disabled_is_a_noop(self):
        x = noise(1.0, seed=2)
        wt = [(0.5, 0.2, "plan")]
        y = E._de_accent(x, FS, wt, np.random.default_rng(0), ["plan"],
                         {"enabled": False, "given_soften_db": -3.0})
        assert np.allclose(y, x)

    def test_only_given_words_touched(self):
        x = noise(1.0, seed=3)
        wt = [(0.2, 0.15, "hello"), (0.6, 0.15, "world")]
        y = E._de_accent(x, FS, wt, np.random.default_rng(0), ["world"],
                         {"enabled": True, "given_soften_db": -4.0})
        # the non-given word keeps its level
        assert abs(rms(y[int(0.2 * FS):int(0.35 * FS)])
                   - rms(x[int(0.2 * FS):int(0.35 * FS)])) < 1e-6


# --------------------------------------------------------------------------- #
# Phase C — emphasis brightness tilt                                            #
# --------------------------------------------------------------------------- #

class TestEmphasisBrightness:
    def test_emphasis_words_get_brighter(self):
        x = noise(1.0, seed=4)
        wt = [(0.5, 0.2, "wow")]
        cfg = {"enabled": True, "tilt_db": 3.0, "hz": 3800.0, "q": 0.9}
        y = E._emphasis_tilt(x, FS, wt, ["wow"], cfg)
        assert hf_energy(y) > hf_energy(x)

    def test_disabled_is_a_noop(self):
        x = noise(1.0, seed=5)
        wt = [(0.5, 0.2, "wow")]
        y = E._emphasis_tilt(x, FS, wt, ["wow"],
                             {"enabled": False, "tilt_db": 3.0})
        assert np.allclose(y, x)


# --------------------------------------------------------------------------- #
# Phase E — thought gaps (and breath sizing)                                    #
# --------------------------------------------------------------------------- #

class TestThoughtGap:
    def test_thought_gap_lengthens_pauses(self):
        cfg = snfy.create_engine().config
        em = E.EMOTIONS["neutral"]
        with_gap = np.mean([
            E._gap_ms("sentence", em, np.random.default_rng(s), cfg)
            for s in range(300)])
        cfg_off = copy.deepcopy(cfg)
        cfg_off["thought_gap"] = {"enabled": False}
        without = np.mean([
            E._gap_ms("sentence", em, np.random.default_rng(s), cfg_off)
            for s in range(300)])
        assert with_gap > without

    def test_breath_size_from_pause_wired(self):
        cfg = snfy.create_engine().config
        assert cfg["breath"]["size_from_pause"]["enabled"] is True


# --------------------------------------------------------------------------- #
# Phase F — room-tone whisper                                                   #
# --------------------------------------------------------------------------- #

class TestWhisperRoomTone:
    def _track(self):
        t = np.arange(int(FS * 0.5)) / FS
        tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float64)
        return np.concatenate([np.zeros(FS // 2), tone, np.zeros(FS // 2)])

    def _cfg(self, enabled):
        base = copy.deepcopy(snfy.create_engine().config)
        base["room_tone"]["enabled"] = False
        base["whisper_room_tone"] = {"level_db": -52.0, "room_ratio": 0.7,
                                     "breath_ratio": 0.3, "enabled": enabled}
        return base

    def test_whisper_gets_a_room_bed(self):
        track = self._track()
        y_on, _ = E.master(track.copy(), FS, E.EMOTIONS["whisper"],
                           E.Persona(), np.random.default_rng(0),
                           self._cfg(True), emotion_key="whisper")
        y_off, _ = E.master(track.copy(), FS, E.EMOTIONS["whisper"],
                            E.Persona(), np.random.default_rng(0),
                            self._cfg(False), emotion_key="whisper")
        assert not np.allclose(y_on, y_off)

    def test_non_whisper_unaffected(self):
        track = self._track()
        y_on, _ = E.master(track.copy(), FS, E.EMOTIONS["neutral"],
                           E.Persona(), np.random.default_rng(0),
                           self._cfg(True), emotion_key="neutral")
        y_off, _ = E.master(track.copy(), FS, E.EMOTIONS["neutral"],
                            E.Persona(), np.random.default_rng(0),
                            self._cfg(False), emotion_key="neutral")
        assert np.allclose(y_on, y_off)


# --------------------------------------------------------------------------- #
# Phase G — verbatim mode (+ the Emotion enum normalisation fix)                #
# --------------------------------------------------------------------------- #

class TestVerbatim:
    def test_numbers_are_not_expanded(self):
        plain = E._preprocess("The price is 200.")
        verb = E._preprocess("The price is 200.", verbatim=True)
        assert "two hundred" in plain.lower()
        assert "200" in verb
        assert "two hundred" not in verb.lower()

    def test_currency_is_not_expanded(self):
        verb = E._preprocess("Pay Rs. 4500 now.", verbatim=True)
        assert "4500" in verb
        assert "rupees" not in verb.lower()

    def test_contractions_are_not_applied(self):
        verb = E._preprocess("I am not ready.", verbatim=True)
        assert "I am" in verb
        assert "I'm" not in verb


class TestEmotionNormalisation:
    def test_segment_accepts_enum(self):
        cfg = snfy.create_engine().config
        segs = E._segment("Hello there, how are you?", snfy.Emotion.HAPPY,
                          E.Persona(), -2, np.random.default_rng(1), config=cfg)
        assert segs
        assert segs[0]["emo"] == "happy"

    def test_segment_accepts_string(self):
        cfg = snfy.create_engine().config
        segs = E._segment("Hello there, how are you?", "happy",
                          E.Persona(), -2, np.random.default_rng(1), config=cfg)
        assert segs[0]["emo"] == "happy"


# --------------------------------------------------------------------------- #
# Phase D — hesitation fillers (naturalness refinements)                        #
# --------------------------------------------------------------------------- #

def _seg(kind="sentence", pause=300):
    return {"text": "word", "kind": kind, "pause_ms": pause, "pitch": 0,
            "rate": "+0%", "volume": "+0%", "emph": [], "key_words": []}


def _fillers(out):
    return [s for s in out if s.get("is_filler")]


class TestHesitationFillers:
    def test_disabled_or_wordless_is_a_noop(self):
        segs = [_seg(), _seg()]
        assert E._insert_fillers(
            segs, {"enabled": True, "probability": 1.0, "words": []},
            np.random.default_rng(0)) is segs

    def test_boundary_kinds_anchor_to_sentences(self):
        cfg = {"enabled": True, "probability": 1.0, "words": ["uhm"],
               "min_pause_ms": 100, "boundary_kinds": ["sentence"],
               "split_jitter": 0.0, "min_spacing_segments": 0}
        out = E._insert_fillers(
            [_seg("sentence", 300), _seg("clause", 300)],
            cfg, np.random.default_rng(0))
        assert len(_fillers(out)) == 1

    def test_min_spacing_prevents_stutter(self):
        cfg = {"enabled": True, "probability": 1.0, "words": ["uhm"],
               "min_pause_ms": 100, "min_spacing_segments": 2}
        tight = [_seg() for _ in range(4)]
        spaced = E._insert_fillers(tight, cfg, np.random.default_rng(0))
        assert len(_fillers(spaced)) == 1
        cfg["min_spacing_segments"] = 0
        dense = E._insert_fillers([_seg() for _ in range(4)], cfg,
                                  np.random.default_rng(0))
        assert len(_fillers(dense)) > 1

    def test_split_jitter_stays_bounded(self):
        cfg = {"enabled": True, "probability": 1.0, "words": ["uhm"],
               "min_pause_ms": 100, "split_jitter": 0.3}
        for seed in range(20):
            out = E._insert_fillers([_seg("sentence", 300), _seg()], cfg,
                                    np.random.default_rng(seed))
            kept = out[0]["pause_ms"]
            assert 0.2 * 300 <= kept <= 0.8 * 300 + 1e-6

    def test_draw_out_slows_and_lengthens(self):
        cfg = {"enabled": True, "probability": 1.0, "words": ["uhm"],
               "min_pause_ms": 100,
               "draw_out": {"enabled": True, "probability": 1.0,
                            "rate": "-14%", "max_seconds": 0.9}}
        out = E._insert_fillers([_seg(), _seg()], cfg,
                                np.random.default_rng(0))
        f = _fillers(out)[0]
        assert f["rate"] == "-14%"
        assert f["filler_max_s"] == 0.9

    def test_no_filler_after_final_segment(self):
        cfg = {"enabled": True, "probability": 1.0, "words": ["uhm"],
               "min_pause_ms": 100}
        out = E._insert_fillers([_seg()], cfg, np.random.default_rng(0))
        assert _fillers(out) == []

    def test_hesitation_sits_inside_the_pause(self):
        cfg = {"enabled": True, "probability": 1.0, "words": ["uhm"],
               "min_pause_ms": 100, "split_jitter": 0.0}
        out = E._insert_fillers([_seg("sentence", 400), _seg()], cfg,
                                np.random.default_rng(0))
        assert out[0]["pause_ms"] + out[1]["pause_ms"] <= 400 + 1e-6
        assert out[1]["is_filler"]


# --------------------------------------------------------------------------- #
# Integration — the whole render path with the phases enabled (needs network)   #
# --------------------------------------------------------------------------- #

def test_render_with_all_phases(tmp_path):
    eng = snfy.SonifyEngine()
    c = copy.deepcopy(eng.config)
    c["humanize"]["clause_takes"]["enabled"] = True
    c["de_accent"]["enabled"] = True
    c["thought_gap"]["enabled"] = True
    eng.config = c
    out = tmp_path / "phases.wav"
    text = ("The plan must be clear, the plan must be fair, "
            "and the plan must be honest.")
    r = eng.synthesize(text, emotion="warm", output_path=str(out), seed=7)
    assert out.exists()
    assert r.duration > 0.3
    assert r.sample_rate == 48000


def test_render_with_hesitations(tmp_path):
    eng = snfy.SonifyEngine()
    c = copy.deepcopy(eng.config)
    c["pause_filler"]["enabled"] = True
    c["pause_filler"]["probability"] = 1.0
    eng.config = c
    out = tmp_path / "hesitations.wav"
    text = ("First, we plan the work. Then, we do the work. "
            "Finally, we review the work.")
    r = eng.synthesize(text, emotion="thoughtful", output_path=str(out), seed=11)
    assert out.exists()
    assert r.duration > 0.3

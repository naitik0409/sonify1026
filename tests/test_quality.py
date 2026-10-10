"""
Sonify 1026 voice engine — quality & regression test suite.

Objective, waveform-derived checks only: this suite is designed to catch the
kinds of defects that were reported by ear (unclear consonants, shivering
between sentences, a word repeated at a seam, clipping, robotic monotone)
without requiring anyone to listen.

Run:
    pytest tests -q
    python tests/test_quality.py          # same suite, standalone runner
"""

import os
import re
import sys
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import (measure, load, is_hindi, silence_runs, energy_dips,  # noqa: E402
                     clarity, pitch_stats, modulation, db)
from snfy import SonifyEngine, Emotion, Language                 # noqa: E402
from snfy.engine import (_analyze_text_dynamics, _pack_utterances,    # noqa: E402
                           _preprocess, EMOTIONS, VOICES)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_pytest")

EN = "Please submit the report by tomorrow afternoon, thank you."
EN_LONG = ("When I was a child, my grandmother used to tell me stories every evening "
           "about the village where she grew up, and those quiet evenings taught me "
           "more about patience than any classroom ever could.")
HI = "कृपया यह रिपोर्ट कल तक जमा कर दें, धन्यवाद।"
HI_LONG = ("जब मैं छोटी थी, मेरी दादी मुझे हर शाम कहानियाँ सुनाती थीं, और वो शामें "
           "मुझे किसी भी कक्षा से ज़्यादा धैर्य सिखाती थीं।")


@pytest.fixture(scope="session")
def engine():
    os.makedirs(OUT, exist_ok=True)
    return SonifyEngine()


def render(engine, text, tag, emotion=Emotion.NEUTRAL, language=None, **kw):
    path = os.path.join(OUT, tag + ".wav")
    engine.synthesize(text, emotion=emotion, language=language,
                      output_path=path, **kw)
    return measure(path)


# =========================================================================== #
# 1. Signal integrity                                                         #
# =========================================================================== #

class TestSignalIntegrity:
    def test_wellformed_audio(self, engine):
        m = render(engine, EN, "sig_en")
        assert m["finite"], "non-finite samples (NaN/Inf) in output"
        assert m["duration"] > 0.3, f"implausibly short: {m['duration']:.2f}s"
        assert abs(m["dc_offset"]) < 1e-3, f"DC offset {m['dc_offset']:.5f}"

    def test_no_clipping_and_ceiling_respected(self, engine):
        cfg = engine.config["persona"]["peak_ceiling_db"]
        for tag, txt in (("clip_en", EN), ("clip_hi", HI)):
            m = render(engine, txt, tag)
            assert m["clipped_samples"] == 0, f"{tag}: {m['clipped_samples']} clipped samples"
            # the limiter must never overshoot its configured ceiling
            assert m["peak_db"] <= cfg + 0.05, \
                f"{tag}: peak {m['peak_db']:.2f} dB exceeds ceiling {cfg} dB"

    def test_sample_rate_matches_config(self, engine):
        want = engine.config["output"]["sample_rate"]
        m = render(engine, EN, "sr_en")
        assert m["sample_rate"] == want, f"{m['sample_rate']} != {want}"

    def test_no_dead_air_or_dropouts(self, engine):
        for tag, txt in (("gap_en", EN_LONG), ("gap_hi", HI_LONG)):
            m = render(engine, txt, tag)
            assert m["max_silence"] < 1.6, \
                f"{tag}: {m['max_silence']:.2f}s of internal silence (dropped audio?)"

    def test_no_splice_discontinuities(self, engine):
        # A bad splice erupts out of silence; a sharp jump inside loud speech
        # is just a natural plosive (Hindi retroflex stops are very sharp).
        for tag, txt in (("jump_en", EN_LONG), ("jump_hi", HI_LONG)):
            m = render(engine, txt, tag)
            assert m["splice_jump"] < 0.55, \
                f"{tag}: splice jump {m['splice_jump']:.3f} (click/splice)"


# =========================================================================== #
# 2. Phonetic clarity                                                         #
# =========================================================================== #

class TestPhoneticClarity:
    def test_no_energy_holes_from_time_domain_processing(self, engine):
        """Unit-level: the OLA stages must not punch holes in the waveform.

        Measured on a synthetic sustained tone — on real speech a deep dip is
        indistinguishable from a stop-closure, so this is checked where the
        ground truth is known.
        """
        from snfy.engine import _apply_pitch_arc, _build_emotion_contour, _ola_expand
        fs = 48000
        n = fs * 2
        t = np.arange(n) / fs
        tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in
                   enumerate([220, 440, 660, 880], start=1))
        tone = (tone / np.max(np.abs(tone)) * 0.5).astype(np.float32)

        for emo in engine.config["pitch_contour"]["emotion_modulation"]:
            contour = _build_emotion_contour(90, emo, "sentence", engine.config)
            y = _apply_pitch_arc(tone, fs, contour)
            assert energy_dips(y, fs) == 0, f"{emo}: pitch arc punched energy holes"
            rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
            base = np.sqrt(np.mean(tone.astype(np.float64) ** 2))
            assert abs(rms / base - 1.0) < 0.05, f"{emo}: arc changed level by {20*np.log10(rms/base):.1f} dB"

        for factor in (1.1, 1.26):
            y = _ola_expand(tone[:int(fs * 0.5)].astype(np.float64), fs, factor)
            assert energy_dips(y, fs) == 0, f"OLA x{factor} punched energy holes"
            assert abs(len(y) / (fs * 0.5) - factor) < 0.05, "OLA expansion wrong length"

    def test_speech_has_no_drops_or_splices(self, engine):
        """On real audio: no dead air and no hard splices."""
        for tag, txt in (("hole_en", EN_LONG), ("hole_hi", HI_LONG)):
            m = render(engine, txt, tag)
            assert m["max_silence"] < 1.6, f"{tag}: {m['max_silence']:.2f}s of silence"
            assert m["splice_jump"] < 0.55, \
                f"{tag}: splice jump {m['splice_jump']:.3f} (click/splice)"

    def test_consonant_bands_present(self, engine):
        """Sibilant/fricative energy must not collapse."""
        for tag, txt in (("cons_en", EN), ("cons_hi", HI)):
            render(engine, txt, tag)
            x, fs = load(os.path.join(OUT, tag + ".wav"))
            c = clarity(x, fs)
            assert c["high"] > 5.0, f"{tag}: 4-9k energy too low ({c['high']:.1f} dB)"
            assert c["air"] > -6.0, f"{tag}: 9-14k energy too low ({c['air']:.1f} dB)"

    def test_output_not_muffled_versus_source(self, engine):
        """Our chain must not dull the neural voice it receives."""
        import asyncio
        import edge_tts
        from snfy.engine import _decode_mp3, _is_hindi

        def pick_voice(text: str) -> str:
            cfg = engine.config
            vm = cfg.get("voice_mode", {})
            if vm.get("mode") == "single":
                vkey = vm.get("primary", "swara")
                return VOICES[vkey]["id"]
            else:
                return VOICES["neerja" if not _is_hindi(text) else "swara"]["id"]

        async def raw(voice_id: str, rate: str = "+0%", pitch: str = "+0Hz"):
            com = edge_tts.Communicate(EN, voice_id, rate=rate, volume="+0%", pitch=pitch)
            buf = bytearray()
            async for ch in com.stream():
                if ch["type"] == "audio":
                    buf += ch["data"]
            return bytes(buf)

        vkey = pick_voice(EN)
        rate = engine.config.get("emotions", {}).get("neutral", {}).get("rate", "+0%")
        pitch = engine.config.get("emotions", {}).get("neutral", {}).get("pitch", 0)
        xr, fsr = _decode_mp3(asyncio.run(raw(vkey, rate=rate, pitch=f"{pitch:+d}Hz")))
        c_raw = clarity(xr.astype(np.float64), fsr)
        m = render(engine, EN, "muffle_en")
        assert m["clarity"] >= c_raw["clarity"] - 1.0, \
            f"clarity dropped vs source: {m['clarity']:.2f} vs {c_raw['clarity']:.2f}"

    def test_loudness_is_consistent(self, engine):
        for tag, txt in (("loud_en", EN), ("loud_hi", HI)):
            m = render(engine, txt, tag)
            assert -30.0 < m["rms_db"] < -14.0, f"{tag}: rms {m['rms_db']:.1f} dB out of range"
            assert m["loudness_range_db"] > 2.0, \
                f"{tag}: dynamic range {m['loudness_range_db']:.1f} dB — sounds compressed/flat"


# =========================================================================== #
# 3. Voice identity                                                           #
# =========================================================================== #

class TestVoiceIdentity:
    def test_pitch_is_female_and_stable(self, engine):
        for tag, txt in (("f0_en", EN_LONG), ("f0_hi", HI_LONG)):
            m = render(engine, txt, tag)
            f0 = m["f0_median"]
            assert f0 > 185.0, f"{tag}: F0 {f0:.1f} Hz reads as male/low"
            assert f0 < 300.0, f"{tag}: F0 {f0:.1f} Hz is unnaturally high"

    def test_no_robotic_monotone(self, engine):
        """Monotone = almost no pitch movement. Real speech moves pitch."""
        for tag, txt in (("mono_en", EN_LONG), ("mono_hi", HI_LONG)):
            m = render(engine, txt, tag)
            span = m["f0_p95"] - m["f0_p05"]
            assert span > 28.0, f"{tag}: pitch span only {span:.1f} Hz — monotone"

    def test_no_excessive_pitch_wobble(self, engine):
        """Shivering shows up as unstable pitch in TYPICAL frames.

        Frame-to-frame pitch change scales with speaking rate, so there is no
        meaningful fixed ceiling on the p95 value; the typical-frame step is
        the stable measure and is compared against the neural source.
        """
        import asyncio
        import edge_tts
        from snfy.engine import _decode_mp3

        async def raw():
            com = edge_tts.Communicate(HI_LONG, "hi-IN-SwaraNeural",
                                       rate="-2%", volume="+0%",
                                       pitch="%+dHz" % engine.config["character"]["base_pitch"])
            buf = bytearray()
            async for ch in com.stream():
                if ch["type"] == "audio":
                    buf += ch["data"]
            return bytes(buf)

        xr, fsr = _decode_mp3(asyncio.run(raw()))
        src = pitch_stats(xr.astype(np.float64), fsr)

        for tag, txt in (("jit_en", EN_LONG), ("jit_hi", HI_LONG)):
            m = render(engine, txt, tag)
            assert m["jitter_median"] < 20.0, \
                f"{tag}: typical pitch step {m['jitter_median']:.1f} Hz — shivering"
            assert m["jitter_median"] <= src["jitter_median"] * 2.5 + 5.0, \
                f"{tag}: pitch step {m['jitter_median']:.1f} Hz vs source " \
                f"{src['jitter_median']:.1f} Hz"
            assert m["jitter_p95"] < 200.0, \
                f"{tag}: pitch jitter p95 {m['jitter_p95']:.0f} Hz — gross instability"

    def test_chest_resonance_present(self, engine):
        """Low-to-mid body energy must not be hollowed out."""
        x, fs = load(os.path.join(OUT, "f0_en.wav"))
        c = clarity(x, fs)
        assert c["low"] > c["body"] - 6.0, \
            f"low-mid {c['low']:.1f} dB far below body {c['body']:.1f} dB — thin/hollow"


# =========================================================================== #
# 4. Text integrity — regression for the repeated-word bug                     #
# =========================================================================== #

def _norm_words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


class TestTextIntegrity:
    @pytest.mark.parametrize("text", [EN_LONG, HI_LONG, EN, HI])
    def test_no_words_lost_or_duplicated(self, engine, text):
        """Regression: takes used to be cut mid-clause, repeating the seam word."""
        import snfy.engine as E

        captured = []
        original = E._edge_one

        async def spy(seg, voice):
            if not seg.get("is_filler"):
                captured.append(seg["text"])
            return await original(seg, voice)

        E._edge_one = spy
        try:
            render(engine, text, "integrity_" + str(abs(hash(text)) % 10 ** 6))
        finally:
            E._edge_one = original

        assert captured, "nothing was sent to the voice backend"
        spoken = _norm_words(" ".join(captured))
        expected = _norm_words(text)
        from collections import Counter
        missing = Counter(expected) - Counter(spoken)
        extra = Counter(spoken) - Counter(expected)
        assert not missing, f"words dropped: {dict(missing)}"
        assert not extra, f"words duplicated: {dict(extra)}"

    @pytest.mark.parametrize("text", [EN_LONG, HI_LONG, EN, HI])
    def test_every_take_ends_on_sentence_punctuation(self, engine, text):
        """Regression: takes ending mid-clause caused seams and shivering."""
        n = _preprocess(text, contractions=True)
        cfg = {"humanize": engine.config["humanize"]}
        for take, kind in _pack_utterances(n, 260, cfg):
            assert take.rstrip().endswith((".", "!", "?", "…", "।")), \
                f"take ends mid-clause: ...{take[-45:]!r}"

    def test_no_trailing_hesitation_after_the_last_sentence(self, engine):
        """Regression: a spoken 'ahm' used to dangle after the final phrase."""
        import snfy.engine as E

        saved = engine.config["pause_filler"]["enabled"]
        saved_max = engine.config["humanize"]["soft_max_utterance_chars"]
        saved_chars = engine.config["humanize"]["max_utterance_chars"]
        engine.config["pause_filler"]["enabled"] = True
        engine.config["pause_filler"]["probability"] = 1.0
        engine.config["pause_filler"]["micro"]["probability"] = 1.0
        # several sentences AND a small take size, so interior pauses exist
        engine.config["humanize"]["max_utterance_chars"] = 40
        engine.config["humanize"]["soft_max_utterance_chars"] = 40
        multi = "Hello there. How are you? I am fine. Thank you. See you soon."
        try:
            for trial in range(6):
                captured = []
                original = E._edge_one

                async def spy(seg, voice):
                    captured.append((bool(seg.get("is_filler")), seg["text"]))
                    return await original(seg, voice)

                E._edge_one = spy
                try:
                    engine.synthesize(multi, emotion=Emotion.WARM,
                                      language=Language.EN_IN,
                                      output_path=os.path.join(OUT, "tail.wav"),
                                      seed=1000 + trial)
                finally:
                    E._edge_one = original

                fillers = [i for i, (f, _) in enumerate(captured) if f]
                assert fillers, "no filler placed even at probability 1.0"
                assert max(fillers) < len(captured) - 1, \
                    "a hesitation was placed after the final segment"
                assert captured[-1][0] is False, \
                    "the last segment spoken is a filler, not speech"
        finally:
            engine.config["pause_filler"]["enabled"] = saved
            engine.config["humanize"]["soft_max_utterance_chars"] = saved_max
            engine.config["humanize"]["max_utterance_chars"] = saved_chars
            engine._apply_config()

    def test_take_kind_matches_its_own_punctuation(self, engine):
        # Force one take per sentence so each kind can be checked.
        n = _preprocess("Hello there! How are you? I am fine.", contractions=True)
        cfg = {"humanize": {"soft_max_utterance_chars": 1}}
        packed = _pack_utterances(n, 1, cfg)
        assert len(packed) >= 3, f"expected 3 takes, got {len(packed)}"
        kinds = {t.rstrip()[-1]: k for t, k in packed}
        assert kinds.get("?") == "question", f"question mislabelled: {kinds}"
        assert kinds.get("!") == "exclaim", f"exclamation mislabelled: {kinds}"
        assert kinds.get(".") == "sentence", f"statement mislabelled: {kinds}"


# =========================================================================== #
# 5. Adaptive prosody                                                          #
# =========================================================================== #

class TestAdaptiveProsody:
    POS = "That is wonderful news, I am so happy and grateful for you!"
    TENDER = "I care about you, and I hope you feel safe and warm here."
    NEG = "This is terrible, I am really scared of what went wrong."
    FLAT = "Please submit the report by tomorrow afternoon."

    def test_valence_is_detected(self, engine):
        cfg = engine.config
        pos = _analyze_text_dynamics(self.POS, cfg)
        neg = _analyze_text_dynamics(self.NEG, cfg)
        ten = _analyze_text_dynamics(self.TENDER, cfg)
        assert pos["smile"] > 0.5, "positive text not detected as smiling"
        assert ten["soften"] > 0.3, "tender text not detected"
        assert neg["valence"] < 0, "negative text not given negative valence"
        assert _analyze_text_dynamics(self.FLAT, cfg)["smile"] < 0.2

    def test_smile_lifts_pitch_and_brightness(self, engine):
        cfg = engine.config
        pos = _analyze_text_dynamics(self.POS, cfg)
        neg = _analyze_text_dynamics(self.NEG, cfg)
        assert pos["pitch_delta"] > neg["pitch_delta"], \
            "positive phrases must lift pitch relative to negative"
        assert pos["bright_delta"] > neg["bright_delta"], \
            "positive phrases must be brighter (smiling)"

    def test_breathing_scales_with_content(self, engine):
        cfg = engine.config
        emph = _analyze_text_dynamics(self.POS, cfg)
        flat = _analyze_text_dynamics(self.FLAT, cfg)
        assert emph["breath_scale"] > flat["breath_scale"], \
            "breathing interval must adapt to the passage"

    def test_pause_tracks_emotional_subtext(self, engine):
        cfg = engine.config
        pos = _analyze_text_dynamics(self.POS, cfg)
        neg = _analyze_text_dynamics(self.NEG, cfg)
        assert pos["pause_scale"] > neg["pause_scale"], \
            "emotional subtext must modulate pause length"

    def test_key_and_hedging_words_identified(self, engine):
        cfg = engine.config
        emph = _analyze_text_dynamics("This is absolutely wonderful and I love it.", cfg)
        hedge = _analyze_text_dynamics("Perhaps we should maybe think about it.", cfg)
        assert emph["key_words"], "content words not detected"
        assert hedge["low_words"], "hedging words not detected"

    def test_delivery_varies_between_scripts(self, engine):
        """Different text must not produce identical prosody targets."""
        import snfy.engine as E
        seen = []

        original = E._edge_one

        async def spy(seg, voice):
            if not seg.get("is_filler"):
                seen.append((seg["text"], seg["pitch"], seg["rate"], seg["volume"]))
            return await original(seg, voice)

        E._edge_one = spy
        try:
            render(engine, self.POS, "adapt_a", seed=7)
            render(engine, self.NEG, "adapt_b", seed=7)
        finally:
            E._edge_one = original

        assert seen, "no segments captured"
        pa, pb = seen[0], seen[1]
        assert (pa[1], pa[2], pa[3]) != (pb[1], pb[2], pb[3]), \
            "positive and negative scripts received identical prosody"

    def test_each_call_is_a_unique_performance(self, engine):
        a = render(engine, EN_LONG, "uniq_a")
        b = render(engine, EN_LONG, "uniq_b")
        assert abs(a["duration"] - b["duration"]) > 0.01 or \
            abs(a["jitter_p95"] - b["jitter_p95"]) > 0.01, \
            "identical text produced an identical performance (rigid voice)"


# =========================================================================== #
# 6. Coverage — every emotion, voice, register, accent                        #
# =========================================================================== #

class TestCoverage:
    def test_every_emotion_synthesises(self, engine):
        for name in EMOTIONS:
            emotion = getattr(Emotion, name.upper(), None)
            assert emotion is not None, f"Emotion.{name.upper()} missing from the enum"
            m = render(engine, "Hello there, this is a test.", f"emo_{name}",
                       emotion=emotion, seed=11)
            assert m["finite"] and m["duration"] > 0.3, f"emotion {name} failed"
            assert m["f0_median"] > 150.0, f"emotion {name} lost pitch ({m['f0_median']:.0f} Hz)"

    def test_every_emotion_in_hindi(self, engine):
        for name in EMOTIONS:
            emotion = getattr(Emotion, name.upper(), None)
            m = render(engine, "नमस्ते, यह एक परीक्षण है।", f"emohi_{name}",
                       emotion=emotion, language=Language.HI_IN, seed=11)
            assert m["finite"] and m["duration"] > 0.3, f"hindi emotion {name} failed"
            assert m["f0_median"] > 150.0, f"hindi emotion {name} lost pitch"

    def test_every_configured_voice_is_reachable(self, engine):
        for vkey in engine.config["voices"]:
            assert vkey in VOICES, f"voice {vkey} not registered"

    def test_registers(self, engine):
        for reg in engine.config["registers"]:
            m = render(engine, EN, f"reg_{reg}", register=reg)
            assert m["finite"] and m["duration"] > 0.3, f"register {reg} failed"

    def test_accents(self, engine):
        for acc in engine.config["accents"]:
            m = render(engine, EN, f"acc_{acc}", accent=acc)
            assert m["finite"] and m["duration"] > 0.3, f"accent {acc} failed"

    def test_hinglish_code_switching(self, engine):
        for tag, txt in (("hing1", "Yaar, kal ki meeting bahut important hai, "
                                   "so on time aa jaana, aur laptop lete aana."),
                         ("hing2", "ठीक है, the point is simple — जो भी काम करो, "
                                   "पूरे मन से करो।")):
            m = render(engine, txt, tag)
            assert m["finite"] and m["duration"] > 0.5, f"{tag} failed"
            assert m["f0_median"] > 185.0, f"{tag}: code-switch F0 {m['f0_median']:.0f} Hz"


# =========================================================================== #
# 7. Edge cases                                                                #
# =========================================================================== #

class TestEdgeCases:
    @pytest.mark.parametrize("text", [
        "Hello.",
        "Hi!",
        "Really?!",
        "Wait...",
        "Hmm.",
        "1234567890.",
        "The meeting is at 3:30 p.m. on 12/05/2026, okay?",
        "Dr. Smith and Mr. Jones went to Washington, D.C. yesterday!",
        "\u0928\u092e\u0938\u094d\u0924\u0947\u0964",
        "\u2026",
        "one two three, four five six, seven eight nine ten.",
        "A" * 400 + ".",
    ])
    def test_degenerate_inputs(self, engine, text):
        m = render(engine, text, "edge_" + str(abs(hash(text)) % 10 ** 6))
        assert m["finite"], f"non-finite output for {text[:20]!r}"
        assert m["peak_db"] <= engine.config["persona"]["peak_ceiling_db"] + 0.05

    def test_very_long_input(self, engine):
        text = " ".join(["This is a sentence in a very long script that keeps going."] * 12)
        m = render(engine, text, "edge_verylong")
        assert m["finite"]
        assert m["duration"] > 20.0, f"long input truncated to {m['duration']:.1f}s"
        assert m["peak_db"] <= engine.config["persona"]["peak_ceiling_db"] + 0.05

    def test_speaking_rate_is_sane(self, engine):
        m = render(engine, EN_LONG, "edge_rate")
        cps = len(EN_LONG) / m["duration"]
        assert 6.0 < cps < 30.0, f"speech rate {cps:.1f} chars/s is unnatural"


# =========================================================================== #
# 8. Engine API                                                               #
# =========================================================================== #

class TestEngineApi:
    def test_result_fields(self, engine):
        r = engine.synthesize(EN, emotion=Emotion.WARM, language=Language.EN_IN,
                              output_path=os.path.join(OUT, "api.wav"))
        assert r.duration > 0.3
        assert r.sample_rate == engine.config["output"]["sample_rate"]
        assert r.text == EN
        assert isinstance(r.emotion, Emotion)

    def test_apply_config_picks_up_changes(self, engine):
        before = engine.persona.base_pitch
        cfg = engine.config
        original = cfg["character"]["base_pitch"]
        try:
            cfg["character"]["base_pitch"] = before - 6
            engine._apply_config()
            assert engine.persona.base_pitch == before - 6, \
                "config change was not applied to the persona"
        finally:
            cfg["character"]["base_pitch"] = original
            engine._apply_config()
        assert engine.persona.base_pitch == before

    def test_reload_config_rereads_the_file(self, engine):
        engine.reload_config()
        assert engine.persona.base_pitch == engine.config["character"]["base_pitch"]

    def test_speed_pitch_volume_controls(self, engine):
        # Pin the seed: unseeded delivery varies run to run.
        base = render(engine, EN, "ctl_base", seed=4242)
        engine.set_speed(1.25)
        fast = render(engine, EN, "ctl_fast", seed=4242)
        engine.set_speed(1.0)
        assert fast["duration"] < base["duration"] * 0.92, \
            f"set_speed(1.25) did not speed up: {base['duration']:.2f}s -> {fast['duration']:.2f}s"

        engine.set_pitch(-15)
        low = render(engine, EN, "ctl_low", seed=4242)
        engine.set_pitch(0)
        assert low["f0_median"] < base["f0_median"] - 5.0, \
            "set_pitch had no audible effect"

    def test_offline_mode_flag_exists(self, engine):
        assert hasattr(engine, "_offline_mode")


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q", "--no-header"]))

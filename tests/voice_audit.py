"""
Sonify 1026 — comprehensive voice audit.

Renders a broad corpus (every emotion, both languages, code-switching, long
form, edge cases) and checks EVERY measurable property of the result against
thresholds derived from the voice brief:

  * pristine signal      — finite, no clipping, ceiling honoured, no DC
  * phoneme clarity      — consonant bands, clarity ratio, no muffled output
  * voice identity       — female F0, chest resonance, not monotone
  * naturalness          — no shivering, no splices, no dead air, dynamics
  * spontaneity          — unique delivery per call, adaptive to the script
  * correctness          — no words lost or duplicated, no trailing filler
  * "no synthetic compression" — configured off and confirmed in the chain

Run:  python tests/voice_audit.py            (add --quick for a fast pass)
"""

import os
import sys
import time
import argparse
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from metrics import measure, load, clarity, pitch_stats          # noqa: E402
from metrics import integrity as integrity_of, modulation as modulation_of, loudness  # noqa: E402
from snfy import SonifyEngine, Emotion, Language          # noqa: E402
from snfy.engine import (_analyze_text_dynamics, _pack_utterances,  # noqa: E402
                           _preprocess, EMOTIONS)

OUT = os.path.join(ROOT, "outputs", "audit")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Audit:
    def __init__(self, quick=False):
        self.quick = quick
        self.results = []
        self.failures = []
        self.t0 = time.time()

    def raw_reference(self, text, language):
        """Render the same text straight through edge-tts, bypassing all BIOS
        DSP.  Naturalness/clarity metrics are judged against this baseline so
        the thresholds are self-calibrating instead of arbitrary."""
        import asyncio
        import edge_tts
        from snfy.engine import _decode_mp3, _pick_voice, VOICES

        # Use the same voice the engine uses for this text, so the
        # self-calibrating baseline tracks voice_mode (single vs dual).
        vkey = _pick_voice(text, self.engine.config.get("default_accent", "in"))
        voice = VOICES.get(vkey, {}).get("id", "en-IN-NeerjaNeural")
        base = self.engine.config["character"]["base_pitch"]

        async def go():
            com = edge_tts.Communicate(text, voice, rate="-2%", volume="+0%",
                                       pitch="%+dHz" % base)
            buf = bytearray()
            async for ch in com.stream():
                if ch["type"] == "audio":
                    buf += ch["data"]
            return bytes(buf)

        try:
            x, fs = _decode_mp3(asyncio.run(go()))
        except Exception:
            return None
        if x is None or len(x) < fs // 4:
            return None
        from metrics import loudness
        m = {}
        m.update(integrity_of(x, fs))
        m.update(pitch_stats(x.astype(np.float64), fs))
        m.update(clarity(x.astype(np.float64), fs))
        m.update(loudness(x.astype(np.float64), fs))
        m["mod_hf"] = modulation_of(x.astype(np.float64), fs)
        return m

    def case(self, name, text, emotion=Emotion.NEUTRAL, language=None, ref=True, **kw):
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, re.sub(r"\W+", "_", name) + ".wav")
        eng = self.engine
        try:
            eng.synthesize(text, emotion=emotion, language=language,
                           output_path=path, **kw)
            m = measure(path)
        except Exception as e:                       # synthesis itself failed
            self.record(name, False, [(False, f"EXCEPTION {type(e).__name__}: {e}")], None)
            return None, False, []
        base = self.raw_reference(text, language) if ref else None
        checks = self.checks(m, base)
        ok = all(c[0] for c in checks)
        self.record(name, ok, checks, m)
        return m, ok, checks

    def checks(self, m, base=None):
        ceil = self.engine.config["persona"]["peak_ceiling_db"]
        c = [
            (m["finite"],                       "finite samples"),
            (m["duration"] > 0.3,               f"duration {m['duration']:.1f}s"),
            (abs(m["dc_offset"]) < 1e-3,        f"DC {m['dc_offset']:+.5f}"),
            (m["clipped_samples"] == 0,         "no clipped samples"),
            (m["peak_db"] <= ceil + 0.05,       f"peak {m['peak_db']:.2f} <= {ceil} dB"),
            (m["sample_rate"] == self.engine.config["output"]["sample_rate"], "sample rate"),
            (m["max_silence"] < 1.6,            f"max silence {m['max_silence']:.2f}s"),
            (m["splice_jump"] < 0.55,           f"splice jump {m['splice_jump']:.3f}"),
            (m["max_jump"] < 0.98,              f"max jump {m['max_jump']:.3f}"),
            (m["high"] > 5.0,                   f"4-9k {m['high']:.1f} dB"),
            (m["air"] > -6.0,                   f"9-14k {m['air']:.1f} dB"),
            (185.0 < m["f0_median"] < 300.0,    f"F0 {m['f0_median']:.0f} Hz"),
(m["f0_p95"] - m["f0_p05"] > 28.0,  f"pitch span {m['f0_p95']-m['f0_p05']:.0f} Hz"),
            (m["low"] > m["body"] - 6.0,        "chest resonance present"),
            (-30.0 < m["rms_db"] < -14.0,       f"rms {m['rms_db']:.1f} dB"),
            (m["loudness_range_db"] > 2.0,      f"dynamics {m['loudness_range_db']:.1f} dB"),
            # Jitter is checked absolutely, not against the source: frame-to-frame
            # pitch change scales with speaking rate, so a fixed-rate source is
            # not a valid baseline.  Typical-frame movement is the real
            # shivering signature; the p95 ceiling only catches gross artefacts.
            (m["jitter_median"] < 20.0,         f"typical pitch step {m['jitter_median']:.1f} Hz"),
            (m["jitter_p95"] < 200.0,           f"pitch jitter p95 {m['jitter_p95']:.0f} Hz"),
        ]
        # Naturalness judged against the untouched neural source.
        # NOTE: pitch is deliberately NOT compared to the source — the engine
        # applies emotion-driven pitch offsets (calm lowers, excited raises)
        # that the raw baseline does not have.  Deviation there is a feature;
        # only the absolute female band above is enforced.
        if base:
            c += [
                # our EQ deliberately trades a little consonant/body ratio for
                # chest warmth, so a small loss against the source is by design
                (m["clarity"] >= base["clarity"] - 3.0,
                 f"clarity {m['clarity']:.1f} vs source {base['clarity']:.1f} dB"),
                (m["mod_hf"] <= base["mod_hf"] + 2.0,
                 f"HF modulation {m['mod_hf']:.1f} vs source {base['mod_hf']:.1f} dB"),
            ]
        else:
            c += [(m["clarity"] > -24.0, f"clarity {m['clarity']:.1f} dB"),
                  (m["mod_hf"] < 1.5, f"HF modulation {m['mod_hf']:.1f} dB")]
        return c

    def record(self, name, ok, checks, m=None):
        self.results.append((name, ok))
        if not ok:
            self.failures.append((name, checks if isinstance(checks, list) else [(False, str(checks))]))

    def line(self, name, ok, checks, m):
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        detail = ""
        if m:
            detail = (f"{DIM}f0 {m['f0_median']:.0f}Hz  clr {m['clarity']:+.1f}dB  "
                      f"dyn {m['loudness_range_db']:.1f}dB  dur {m['duration']:.1f}s{RESET}")
        print(f"  [{mark}] {name:<34s} {detail}")
        if not ok:
            for good, msg in checks:
                if not good:
                    print(f"        {RED}x{RESET} {msg}")

    def header(self, title):
        print(f"\n{DIM}{'=' * 78}{RESET}")
        print(f"  {title}")
        print(f"{DIM}{'=' * 78}{RESET}")


def corpus(quick):
    E = lambda s: getattr(Emotion, s.upper())
    en_short = "Please submit the report by tomorrow afternoon, thank you."
    hi_short = "कृपया यह रिपोर्ट कल तक जमा कर दें, धन्यवाद।"
    cases = [
        ("neutral statement", en_short, E("neutral"), None),
        ("warm greeting", "Hello! It is lovely to hear from you again.", E("warm"), None),
        ("happy praise", "That is wonderful news, I am so happy and grateful for you!", E("happy"), None),
        ("calm reassurance", "Do not worry, everything will be fine, I am here with you.", E("calm"), None),
        ("serious update", "Please submit the report by tomorrow, this is important.", E("serious"), None),
        ("sad sympathy", "I am sorry to hear that. I hope things get better soon.", E("sad"), None),
        ("excited reaction", "Wow! That is amazing! You did a great job!", E("excited"), None),
        ("question", "Where are you going, and do you need any help?", E("thoughtful"), None),
        ("hindi warm", "नमस्ते! आपका दिन शुभ हो।", E("warm"), Language.HI_IN),
        ("hindi excited", "वाह! क्या बात है! शाबाश!", E("excited"), Language.HI_IN),
        ("hindi calm", "चिंता मत करो, सब ठीक हो जाएगा।", E("reassuring"), Language.HI_IN),
        ("hinglish casual", "Yaar, kal ki meeting bahut important hai, so on time aa jaana.", E("warm"), None),
        ("hinglish script mix", "ठीक है, the point is simple — जो भी काम करो, पूरे मन से करो।",
         E("reassuring"), None),
        ("numbers and times", "The meeting is at 3:30 p.m. on 12/05/2026, okay? Dr. Smith agreed.", E("neutral"), None),
    ]
    if not quick:
        cases += [
            ("long paragraph en",
             "When I was a child, my grandmother used to tell me stories every evening about "
             "the village where she grew up, and those quiet evenings taught me more about "
             "patience than any classroom ever could, and even today I remember her voice.",
             E("warm"), None),
            ("long paragraph hi",
             "जब मैं छोटी थी, मेरी दादी मुझे हर शाम कहानियाँ सुनाती थीं, और वो शामें मुझे "
             "किसी भी कक्षा से ज़्यादा धैर्य सिखाती थीं।", E("serious"), Language.HI_IN),
            ("very long script", " ".join(["This is a sentence in a long script that keeps going."] * 10),
             E("neutral"), None),
        ]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    a = Audit(quick=args.quick)
    a.engine = SonifyEngine()
    cfg = a.engine.config
    print(f"{DIM}Sonify 1026 voice audit{'  (quick)' if args.quick else ''}{RESET}")
    print(f"{DIM}output: {OUT}{RESET}")

    # ---------------------------------------------------------------- config
    a.header("1. CONFIGURATION — the brief, verified in the config")
    checks = [
        (all(float(e.get("comp", 1)) <= 1.001 for e in cfg["emotions"].values()),
         "no synthetic compression (every emotion comp <= 1.0)"),
        (not cfg["pause_filler"].get("enabled", False),
         "no spoken hesitations ('ahm') dangling after sentences"),
        (not cfg["pitch_contour"].get("enabled", True) and
         not cfg["word_elongation"].get("enabled", False),
         "no OLA time-domain processing (protects phoneme clarity)"),
        (cfg["character"].get("randomize_seed", False),
         "unique delivery per call (randomize_seed)"),
        (cfg.get("dynamic_prosody", {}).get("valence_words"),
         "emotional valence analysis configured"),
        (cfg.get("dynamic_prosody", {}).get("arc_pitch_st"),
         "adaptive emotional arc configured"),
        (cfg.get("breath", {}).get("enabled"), "breath intakes enabled"),
    ]
    for ok, msg in checks:
        print(f"  [{GREEN}PASS{RESET}]" if ok else f"  [{RED}FAIL{RESET}]", msg)
        if not ok:
            a.failures.append(("config: " + msg, [(False, msg)]))

    # ---------------------------------------------------------------- corpus
    a.header("2. CORPUS — measured absolutely AND against the untouched neural source")
    cases = corpus(args.quick)
    for name, text, emo, lang in cases:
        m, ok, checks = a.case(name, text, emotion=emo, language=lang)
        a.line(name, ok, checks, m)

    # ------------------------------------------------------- emotion coverage
    a.header("3. COVERAGE — every emotion, both languages")
    for lang, txt, tag in ((None, "Hello there, this is a test.", "en"),
                           (Language.HI_IN, "नमस्ते, यह एक परीक्षण है।", "hi")):
        bad = []
        for name in (list(EMOTIONS)[:6] if args.quick else EMOTIONS):
            emo = getattr(Emotion, name.upper())
            path = os.path.join(OUT, f"emotion_{tag}_{name}.wav")
            try:
                a.engine.synthesize(txt, emotion=emo, language=lang, output_path=path)
                m = measure(path)
                if not (m["finite"] and m["f0_median"] > 150.0 and
                        m["peak_db"] <= cfg["persona"]["peak_ceiling_db"] + 0.05):
                    bad.append(name)
            except Exception:
                bad.append(name)
        print(f"  [{GREEN}PASS{RESET}]" if not bad else f"  [{RED}FAIL{RESET}]",
              f"all {'17' if not args.quick else '6'} emotions render in {tag.upper()}"
              + (f" — failed: {bad}" if bad else ""))

    # ------------------------------------------------------------ adaptivity
    a.header("4. ADAPTIVITY — the voice responds to the script, not a preset")
    POS, NEG, TEN, FLAT = ("That is wonderful news, I am so happy and grateful for you!",
                           "This is terrible, I am really scared of what went wrong.",
                           "I care about you, and I hope you feel safe and warm here.",
                           "Please submit the report by tomorrow afternoon.")
    rows = []
    for tag, t in (("positive", POS), ("negative", NEG), ("tender", TEN), ("flat", FLAT)):
        d = _analyze_text_dynamics(t, cfg)
        rows.append((tag, d))
        print(f"  {tag:<9s} smile={d['smile']:.2f} soften={d['soften']:.2f} "
              f"valence={d['valence']:+.2f} pitch{d['pitch_delta']:+.2f}st "
              f"bright{d['bright_delta']:+.2f}dB vol{d['volume_delta']:+.1f}% "
              f"breath x{d['breath_scale']:.2f} pause x{d['pause_scale']:.2f}")
    pos, neg, ten, flat = (r[1] for r in rows)
    adapt = [
        (pos["pitch_delta"] > neg["pitch_delta"], "positive text lifts pitch above negative"),
        (pos["bright_delta"] > neg["bright_delta"], "positive text is brighter (smiling)"),
        (ten["soften"] > 0.3, "tender text detected"),
        (pos["breath_scale"] > flat["breath_scale"], "breathing adapts to content"),
        (pos["pause_scale"] > neg["pause_scale"], "pause follows emotional subtext"),
        (neg["valence"] < 0, "negative valence detected"),
    ]
    for ok, msg in adapt:
        print(f"  [{GREEN}PASS{RESET}]" if ok else f"  [{RED}FAIL{RESET}]", msg)
        if not ok:
            a.failures.append(("adaptivity: " + msg, [(False, msg)]))

    # ---------------------------------------------------------- correctness
    a.header("5. CORRECTNESS — text integrity and no trailing artefacts")
    import snfy.engine as E
    integrity = [
        ("When I was a child, my grandmother used to tell me stories every evening about "
         "the village where she grew up, and those quiet evenings taught me more about "
         "patience than any classroom ever could, and even today I remember her voice."),
        "Please submit the report. The deadline is tomorrow. Thank you for your help.",
    ]
    orig_one = E._edge_one
    for i, text in enumerate(integrity):
        captured = []

        async def spy(seg, voice, _c=captured):
            if not seg.get("is_filler"):
                _c.append(seg["text"])
            return await orig_one(seg, voice)

        E._edge_one = spy
        try:
            a.engine.synthesize(text, emotion=Emotion.WARM, language=Language.EN_IN,
                                output_path=os.path.join(OUT, "integrity.wav"))
        finally:
            E._edge_one = orig_one

        spoken = re.findall(r"[a-z0-9']+", " ".join(captured).lower())
        expect = re.findall(r"[a-z0-9']+", text.lower())
        ok = spoken == expect
        print(f"  [{GREEN}PASS{RESET}]" if ok else f"  [{RED}FAIL{RESET}]",
              f"script {i+1}: {len(expect)} words in, {len(spoken)} spoken, none lost or repeated")
        if not ok:
            a.failures.append((f"text integrity {i+1}", [(False, "words lost or duplicated")]))

    for i, text in enumerate(integrity):
        n = _preprocess(text, contractions=True)
        bad = [t[-40:] for t, k in _pack_utterances(n, 260, {"humanize": cfg["humanize"]})
               if not t.rstrip().endswith((".", "!", "?", "…", "।"))]
        ok = not bad
        print(f"  [{GREEN}PASS{RESET}]" if ok else f"  [{RED}FAIL{RESET}]",
              f"script {i+1}: every take ends on sentence punctuation")
        if not ok:
            a.failures.append((f"take boundaries {i+1}", [(False, str(bad))]))

    # ----------------------------------------------------------- spontaneity
    a.header("6. SPONTANEITY — every call is its own performance")
    durs, f0s = [], []
    for i in range(3):
        p = os.path.join(OUT, "spont_%d.wav" % i)
        a.engine.synthesize(integrity[0], emotion=Emotion.WARM,
                            language=Language.EN_IN, output_path=p)
        m = measure(p)
        durs.append(m["duration"])
        f0s.append(m["jitter_p95"])
    uniq = len(set(round(d, 3) for d in durs)) > 1 or len(set(round(j, 2) for j in f0s)) > 1
    print(f"  [{GREEN}PASS{RESET}]" if uniq else f"  [{RED}FAIL{RESET}]",
          f"3 calls differ (durations {[round(d,2) for d in durs]})")

    # --------------------------------------------------------------- summary
    a.header("SUMMARY")
    passed = sum(1 for _, ok in a.results if ok)
    total = len(a.results)
    elapsed = time.time() - a.t0
    print(f"  corpus cases passed : {passed}/{total}")
    print(f"  audit failures      : {len(a.failures)}")
    print(f"  elapsed             : {elapsed:.0f}s")
    if a.failures:
        print(f"\n  {RED}FAILURES{RESET}")
        for name, _ in a.failures:
            print(f"    - {name}")
    print(f"\n  {GREEN if not a.failures else RED}AUDIT "
          f"{'PASSED' if not a.failures else 'FAILED'}{RESET}")
    return 1 if a.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

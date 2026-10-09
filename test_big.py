"""
Sonify 1026 - Long sentence / paragraph stress test.
Checks: synthesis succeeds, duration scales with text, no clipping,
no dead silence in the middle, pitch stays female (F0 >= 185 Hz).
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from snfy import SonifyEngine, Emotion, Language


def f0_median(x, fs):
    x = x - np.mean(x)
    if np.sqrt(np.mean(x ** 2)) < 0.005:
        return 0.0
    f0min, f0max = 70.0, 350.0
    win = int(fs * 0.04)
    hop = int(fs * 0.01)
    lags = np.arange(int(fs / f0max), int(fs / f0min))
    f0s = []
    for s in range(0, max(1, len(x) - win), hop):
        fr = x[s:s + win]
        if len(fr) < win or np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        fr = fr * np.hanning(len(fr))
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        ac = ac / (ac[0] + 1e-12)
        peak = lags[np.argmax(ac[lags])]
        if ac[peak] > 0.35:
            f0s.append(fs / peak)
    return float(np.median(f0s)) if f0s else 0.0


def longest_gap_seconds(x, fs, thresh_db=-55.0):
    """Longest run of near-silence inside the audio (dead air / dropouts)."""
    thr = 10.0 ** (thresh_db / 20.0)
    hop = int(fs * 0.01)
    quiet = []
    for s in range(0, len(x) - hop, hop):
        quiet.append(np.sqrt(np.mean(x[s:s + hop] ** 2)) < thr)
    best = cur = 0
    for q in quiet:
        cur = cur + 1 if q else 0
        best = max(best, cur)
    return best * 0.01


def analyze(path):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return {
        "duration": round(float(len(x) / fs), 3),
        "peak_db": round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 2),
        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 2),
        "f0": round(float(f0_median(x, fs)), 1),
        "max_gap": round(float(longest_gap_seconds(x, fs)), 2),
    }


EN_LONG = [
    ("en_long_sentence_1",
     "Good morning everyone, and thank you for joining us today; before we begin, "
     "I would like to take a few minutes to walk you through the agenda, explain the "
     "key decisions we made last quarter, and then open the floor for any questions "
     "or suggestions that any of you might have, so please keep your notes ready.",
     Emotion.SERIOUS),
    ("en_long_paragraph_1",
     "When I was a child, my grandmother used to tell me stories every evening about "
     "the village where she grew up, about the river that ran behind her house, about "
     "the mango tree that gave shade to the whole street, and about the festivals that "
     "brought every family together. I did not realise it then, but those quiet evenings "
     "taught me more about patience and kindness than any classroom ever could, and even "
     "today, whenever I close my eyes, I can still hear her voice and smell the old "
     "kitchen where she used to sing while making tea for everyone.",
     Emotion.WARM),
    ("en_long_question",
     "Could someone please explain to me how this is supposed to work in practice, "
     "because from where I am sitting, the plan looks incomplete, the timeline looks "
     "unrealistic, and honestly, I am not sure that the team has enough resources to "
     "deliver everything by the end of the month, so what exactly are we agreeing to here?",
     Emotion.THOUGHTFUL),
    ("en_long_excited",
     "This is absolutely wonderful news! I cannot believe that we actually did it after "
     "so many months of hard work, sleepless nights, endless debugging, and so many "
     "failures that I honestly thought we would never get here, but today we finally "
     "made it and I could not be more proud of this entire team!",
     Emotion.EXCITED),
    ("en_long_sad",
     "I still remember the last conversation we had, the words we said to each other, "
     "and the things we promised but never got the chance to fulfil, and sometimes when "
     "the house is completely quiet, I find myself sitting by the window, thinking about "
     "how different everything could have been if we had just spoken a little more honestly.",
     Emotion.SAD),
]

HI_LONG = [
    ("hi_long_sentence_1",
     "नमस्ते दोस्तों, आज के इस कार्यक्रम में आप सभी का स्वागत है, "
     "मैं आपको बताना चाहती हूँ कि पिछले कुछ महीनों में हमारी टीम ने "
     "कितनी मेहनत की है और अब हम अगले चरण की ओर बढ़ रहे हैं।",
     Emotion.WARM),
    ("hi_long_paragraph_1",
     "जब मैं छोटी थी, मेरी दादी मुझे हर शाम कहानियाँ सुनाया करती थीं, "
     "उस गाँव के बारे में जहाँ वो बड़ी हुईं, उस नदी के बारे में जो उनके घर के "
     "पीछे बहती थी, उस आम के पेड़ के बारे में जो पूरी गली को छाया देता था, और "
     "उन त्योहारों के बारे में जो हर परिवार को एक साथ लाते थे। तब मुझे एहसास "
     "नहीं था, लेकिन वो शांत शामें मुझे किसी भी कक्षा से ज़्यादा धैर्य और "
     "दयालुता के बारे में सिखाती थीं।",
     Emotion.SERIOUS),
    ("hi_long_question",
     "क्या कोई मुझे बता सकता है कि यह योजना व्यवहार में आख़िर काम कैसे करने "
     "वाली है, क्योंकि जहाँ तक मैं देख पा रही हूँ, समयसीमा बहुत कम है, "
     "संसाधन भी पर्याप्त नहीं हैं, और मुझे यक़ीन नहीं है कि टीम इतने कम समय "
     "में सब कुछ पूरा कर पाएगी, तो हम ठीक यही तय कर रहे हैं?",
     Emotion.THOUGHTFUL),
    ("hi_long_excited",
     "यह बहुत ही शानदार ख़बर है! मुझे विश्वास ही नहीं हो रहा कि हमने आख़िरकार "
     "यह कर दिखाया, इतने महीनों की कड़ी मेहनत, रातों की नींद उड़ जाने, न ख़त्म "
     "होने वाली नाकामियों के बाद भी, आज हम यहाँ पहुँच गए और मुझे पूरी टीम पर "
     "बहुत गर्व है!",
     Emotion.EXCITED),
    ("hi_long_sad",
     "मुझे अब भी वो आख़िरी बातचीत याद है, वो शब्द जो हमने एक दूसरे से कहे थे, "
     "और वो वादे जो हम पूरे नहीं कर पाए, और कभी कभी जब घर में पूरी तरह "
     "ख़ामोशी छा जाती है, मैं खिड़की के पास बैठकर सोचती रहती हूँ कि सब कुछ "
     "कितना अलग हो सकता था अगर हमने थोड़ा और ईमानदारी से बात की होती।",
     Emotion.SAD),
]

MIXED = [
    ("mixed_long_bi",
     "Hello everyone! Today we are going to talk about a very important topic. "
     "आज हम एक बहुत महत्वपूर्ण विषय पर चर्चा करने वाले हैं, and I promise you "
     "that by the end of this session, आपको सब कुछ समझ आ जाएगा and we will "
     "answer every single question that comes up.",
     Emotion.NEUTRAL),
]

# Hinglish code-switching (Latin-script Hindi + English in one breath)
HINGLISH = [
    ("hinglish_casual",
     "Yaar, kal ki meeting bahut important hai, so on time aa jaana, "
     "aur haan, laptop charger lete aana, warna battery low ho jayegi.",
     Emotion.WARM),
    ("hinglish_professional",
     "Actually, the client ne last night email kiya hai, and he wants the "
     "revised deck by tomorrow morning, so please review it once and share "
     "your feedback before lunch.",
     Emotion.SERIOUS),
    ("hinglish_question",
     "Bhai, project ka status kya hai? Deadline toh Friday hai, "
     "team ready hai na? Sab kuch plan ke according hi chal raha hai, right?",
     Emotion.THOUGHTFUL),
    ("hinglish_script_mix",
     "देखो, the point is simple — जो भी काम करो, पूरे मन से करो, "
     "because half-hearted effort से कुछ नहीं होता.",
     Emotion.REASSURING),
]

CASES = EN_LONG + HI_LONG + MIXED + HINGLISH


def main():
    engine = SonifyEngine()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/big_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 74)
    print("Sonify 1026 LONG SENTENCE / PARAGRAPH TEST")
    print("=" * 74)

    results = []
    passed = 0

    for i, (name, text, emotion) in enumerate(CASES, 1):
        lang = Language.HI_IN if any("ऀ" <= c <= "ॿ" for c in text) else Language.EN_IN
        print(f"\n[{i}] {name}  ({len(text)} chars, {emotion.value})")
        entry = {"name": name, "chars": len(text), "emotion": emotion.value,
                 "language": lang.value if hasattr(lang, "value") else str(lang)}
        try:
            t0 = time.time()
            result = engine.synthesize(
                text=text, emotion=emotion, language=lang,
                output_path=os.path.join(output_dir, f"{i:02d}_{name}.wav"))
            info = analyze(os.path.join(output_dir, f"{i:02d}_{name}.wav"))
            info["synth_sec"] = round(time.time() - t0, 2)

            problems = []
            chars_per_sec = len(text) / max(info["duration"], 0.001)
            if not (1.5 <= chars_per_sec <= 45.0):
                problems.append(f"rate abnormal {chars_per_sec:.1f} chars/s")
            if info["peak_db"] > -1.0:
                problems.append(f"clipping peak {info['peak_db']} dB")
            if info["peak_db"] < -45.0:
                problems.append(f"too quiet peak {info['peak_db']} dB")
            if info["max_gap"] > 1.5:
                problems.append(f"dead gap {info['max_gap']}s")
            if info["f0"] and info["f0"] < 185.0:
                problems.append(f"pitch male/low {info['f0']} Hz")

            entry.update(info)
            entry["problems"] = problems
            entry["status"] = "PASS" if not problems else "WARN"
            if not problems:
                passed += 1
            print(f"    {entry['status']}: {info['duration']:.1f}s audio, "
                  f"peak {info['peak_db']} dB, F0 {info['f0']} Hz, "
                  f"gap {info['max_gap']}s, {chars_per_sec:.1f} chars/s, "
                  f"synth {info['synth_sec']}s")
            if problems:
                print(f"    -> {', '.join(problems)}")
        except Exception as e:
            entry["status"] = "FAIL"
            entry["error"] = str(e)
            print(f"    FAIL: {e}")
        results.append(entry)

    with open(os.path.join(output_dir, "big_test_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "total": len(CASES), "passed": passed,
                   "results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print(f"RESULTS: {passed}/{len(CASES)} long-text tests clean")
    print(f"Output directory: {output_dir}")
    print("=" * 74)


if __name__ == "__main__":
    main()

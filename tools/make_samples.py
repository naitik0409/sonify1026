"""Render the short MP3 clips used in the README / demo.

Run from the project root:

    python tools/make_samples.py

Produces assets/samples/*.mp3 plus assets/samples/manifest.json.
Requires an internet connection (audio is served by edge-tts).
"""

from __future__ import annotations

import json
import os
import tempfile

import soundfile as sf

import snfy

OUT_DIR = os.path.join("assets", "samples")

CLIPS = [
    {
        "file": "01_neutral.mp3",
        "emotion": "neutral",
        "language": "auto",
        "text": "This is a neutral sample from the Sonify voice engine.",
    },
    {
        "file": "02_happy.mp3",
        "emotion": "happy",
        "language": "auto",
        "text": "That is wonderful news - I am so happy for you!",
    },
    {
        "file": "03_warm.mp3",
        "emotion": "warm",
        "language": "auto",
        "text": "Take your time. There is no rush at all.",
    },
    {
        "file": "04_whisper.mp3",
        "emotion": "whisper",
        "language": "auto",
        "text": "Come closer, I will tell you a secret.",
    },
    {
        "file": "05_hinglish.mp3",
        "emotion": "neutral",
        "language": "auto",
        "text": "Yaar, kal ki meeting bahut important hai.",
    },
    {
        "file": "06_hindi.mp3",
        "emotion": "warm",
        "language": "auto",
        "text": "\u0920\u0940\u0915 \u0939\u0948, \u091c\u094b \u0915\u0930\u094b \u092a\u0942\u0930\u0947 \u092e\u0928 \u0938\u0947 \u0915\u0930\u094b\u0964",
    },
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = snfy.SonifyEngine()
    manifest = []

    tmp = tempfile.mkdtemp(prefix="snfy_samples_")
    for clip in CLIPS:
        wav = os.path.join(tmp, clip["file"] + ".wav")
        result = engine.synthesize(
            clip["text"],
            emotion=clip["emotion"],
            output_path=wav,
        )
        mp3 = os.path.join(OUT_DIR, clip["file"])
        sf.write(mp3, result.audio, result.sample_rate)
        size_kb = os.path.getsize(mp3) // 1024
        print(f"  {clip['file']:>16}  {result.duration:5.2f}s  {size_kb:4d} KB")
        manifest.append(
            {
                "file": clip["file"],
                "emotion": clip["emotion"],
                "language": clip["language"],
                "text": clip["text"],
                "duration": round(result.duration, 2),
            }
        )

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(manifest)} clips to {OUT_DIR}")


if __name__ == "__main__":
    main()

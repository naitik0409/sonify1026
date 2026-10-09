"""
Sonify 1026 CLI - Run as: snfy "Hello world", or python -m snfy "Hello world"
"""

import argparse
import sys
from .engine import SonifyEngine, Emotion


def main():
    parser = argparse.ArgumentParser(description="Sonify 1026 Voice Engine CLI")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--emotion", "-e", default="neutral", help="Emotion preset")
    parser.add_argument("--accent", "-a", default="in", choices=["in", "us"], help="Accent")
    parser.add_argument("--output", "-o", default="output.wav", help="Output file path")
    parser.add_argument("--play", "-p", action="store_true", help="Play audio after synthesis")
    args = parser.parse_args()

    try:
        emotion = Emotion(args.emotion.lower())
    except ValueError:
        print(f"Unknown emotion: {args.emotion}. Using neutral.")
        emotion = Emotion.NEUTRAL

    engine = SonifyEngine()
    result = engine.synthesize(args.text, emotion=emotion, accent=args.accent, output_path=args.output)
    print(f"Saved to {args.output} ({result.duration:.2f}s)")

    if args.play:
        try:
            import sounddevice as sd
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except ImportError:
            print("Install sounddevice to play audio: pip install sounddevice")


if __name__ == "__main__":
    main()

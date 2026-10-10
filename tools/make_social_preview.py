"""Build assets/social-preview.png (1280x640) for the GitHub repo card.

    python tools/make_social_preview.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SAMPLES = os.path.join("assets", "samples")
OUT = os.path.join("assets", "social-preview.png")
W, H = 1280, 640

BG = "#1b1030"
FG = "#f4ecff"
ACCENT = "#ff77c8"
ACCENT2 = "#8a7bff"
DIM = "#b9a9d9"


def main() -> None:
    wav = os.path.join(SAMPLES, "02_happy.mp3")
    audio, _ = sf.read(wav, dtype="float32")
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    step = max(1, len(mono) // 1600)
    env = np.abs(mono[::step])
    if env.max() > 0:
        env = env / env.max()
    t = np.linspace(0, 1, len(env))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.fill_between(t, 0.58, 0.58 + env * 0.16, color=ACCENT2, alpha=0.55, lw=0)
    ax.plot(t, 0.58 + env * 0.16, color=ACCENT, lw=1.0)

    ax.text(0.06, 0.80, "Sonify 1026", color=FG, fontsize=52, fontweight="bold", va="center")
    ax.text(
        0.062, 0.70,
        "Expressive Indian female TTS",
        color=ACCENT, fontsize=22, fontweight="bold", va="center",
    )
    ax.text(
        0.062, 0.635,
        "Adaptive emotion and prosody  ·  English · Hindi · Hinglish",
        color=DIM, fontsize=15, va="center",
    )

    ax.text(0.06, 0.34, "pip install snfy", color=FG, fontsize=20, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a1b47", edgecolor=ACCENT, lw=1.5),
            va="center")

    ax.text(0.94, 0.10, "github.com/naitik0409/sonify1026", color=DIM, fontsize=13,
            ha="right", va="center")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {OUT}  ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()

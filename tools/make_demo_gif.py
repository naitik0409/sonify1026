"""Build assets/demo.gif from the sample clips.

A short looping animation: a playhead sweeps each sample's waveform while the
emotion and text are shown, cycling through all clips. Pure matplotlib - no
screen recording needed.

    python tools/make_demo_gif.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.animation import PillowWriter

SAMPLES = os.path.join("assets", "samples")
OUT = os.path.join("assets", "demo.gif")

BG = "#1b1030"
FG = "#f4ecff"
ACCENT = "#ff77c8"
DIM = "#5b4a7a"
FRAMES_PER_CLIP = 16
FPS = 14


def envelope(audio: np.ndarray, points: int = 600) -> np.ndarray:
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    if len(mono) == 0:
        return np.zeros(points)
    win = max(1, len(mono) // points)
    trimmed = mono[: (len(mono) // win) * win]
    env = np.abs(trimmed).reshape(-1, win).max(axis=1)
    if env.max() > 0:
        env = env / env.max()
    if len(env) < points:
        env = np.pad(env, (0, points - len(env)))
    return env[:points]


def main() -> None:
    manifest = json.load(open(os.path.join(SAMPLES, "manifest.json"), encoding="utf-8"))
    clips = []
    for item in manifest:
        audio, _ = sf.read(os.path.join(SAMPLES, item["file"]), dtype="float32")
        clips.append({"meta": item, "env": envelope(audio)})

    x = np.linspace(0, 1, len(clips[0]["env"]))
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.fill_between([], [], [], color=DIM, alpha=0.0)
    (ghost,) = ax.plot(x, clips[0]["env"], color=DIM, lw=1.0)
    (filled,) = ax.plot([], [], color=ACCENT, lw=1.6)
    (head,) = ax.plot([], [], color=FG, lw=2.0)
    ax.text(0.0, 1.06, "Sonify 1026", color=FG, fontsize=15, fontweight="bold", va="bottom")
    ax.text(
        1.0, 1.07, "pip install snfy", color=DIM, fontsize=9, ha="right", va="bottom"
    )
    emotion_txt = ax.text(0.0, 0.93, "", color=ACCENT, fontsize=12, fontweight="bold", va="top")
    body_txt = ax.text(0.0, 0.16, "", color=FG, fontsize=9, va="top")

    n_frames = 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PillowWriter(fps=FPS).saving(fig, OUT, dpi=100) as writer:
        for clip in clips:
            env = clip["env"]
            meta = clip["meta"]
            ghost.set_ydata(env)
            emotion_txt.set_text(meta["emotion"].capitalize())
            body_txt.set_text("\u201c" + meta["text"] + "\u201d")
            for i in range(FRAMES_PER_CLIP):
                p = (i + 1) / FRAMES_PER_CLIP
                n = max(1, int(p * len(env)))
                filled.set_xdata(x[:n])
                filled.set_ydata(env[:n])
                head.set_xdata([x[n - 1], x[n - 1]])
                head.set_ydata([0, env[n - 1]])
                writer.grab_frame(facecolor=BG)
                n_frames += 1
    plt.close(fig)
    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f"Wrote {OUT}  ({n_frames} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

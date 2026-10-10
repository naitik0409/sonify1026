"""
Sonify 1026 — shared measurement helpers for the test suite.

Every metric here is objective (derived from the rendered waveform) so that
voice-quality regressions are caught automatically instead of relying on
subjective listening alone.
"""

import re
import numpy as np
import soundfile as sf

DEV = re.compile(r"[\u0900-\u097F]")


def is_hindi(t):
    return bool(DEV.search(t))


def load(path):
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float64), fs


def db(v):
    return 20.0 * np.log10(np.asarray(v, dtype=np.float64) + 1e-12)


# --------------------------------------------------------------------------- #
# Basic signal integrity                                                        #
# --------------------------------------------------------------------------- #

def integrity(x, fs):
    """Structural checks: finite, no DC, no clipping, sane length."""
    r = {}
    r["finite"] = bool(np.all(np.isfinite(x)))
    r["peak_db"] = db(np.max(np.abs(x))) if len(x) else -999.0
    r["dc_offset"] = float(np.mean(x))
    r["duration"] = len(x) / fs
    r["clipped_samples"] = int(np.sum(np.abs(x) >= 0.999))
    # max sample-to-sample jump: catches splices/click artefacts
    if len(x) > 1:
        r["max_jump"] = float(np.max(np.abs(np.diff(x))))
    else:
        r["max_jump"] = 0.0
    # splice signature: a large jump that erupts out of near-silence.
    # (A large jump inside loud speech is usually just a natural plosive —
    # Hindi retroflex stops are particularly sharp — not a defect.)
    r["splice_jump"] = splice_jump(x, fs)
    return r


def splice_jump(x, fs, quiet_db=-40.0, window_ms=30.0):
    """Largest sample-to-sample jump that erupts out of near-silence.

    This is the signature of a bad splice or a crossfade click.  Jumps that
    occur inside loud, continuous speech are natural plosive bursts and are
    ignored here.
    """
    if len(x) < 3:
        return 0.0
    d = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    w = max(1, int(fs * window_ms / 1000.0))
    thr = 10.0 ** (quiet_db / 20.0)
    # RMS of the window before each jump, via cumulative energy
    e = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(len(d))
    lo = np.maximum(0, idx - w)
    rms = np.sqrt((e[idx] - e[lo]) / np.maximum(idx - lo, 1) + 1e-20)
    mask = rms < thr
    return float(np.max(d[mask])) if np.any(mask) else 0.0


def envelope_db(x, fs, win_ms=20.0):
    h = max(1, int(fs * win_ms / 1000.0))
    n = len(x) // h
    if n == 0:
        return np.array([])
    e = np.array([np.sqrt(np.mean(x[i * h:(i + 1) * h] ** 2)) for i in range(n)])
    return db(e)


def silence_runs(x, fs, thr_db=-50.0, min_ms=80.0):
    """Durations (s) of internal silence runs — finds dead air / dropped audio."""
    h = max(1, int(fs * 0.01))
    thr = 10.0 ** (thr_db / 20.0)
    q = [np.sqrt(np.mean(x[i:i + h] ** 2)) < thr for i in range(0, len(x) - h, h)]
    runs, cur = [], 0
    for v in q:
        if v:
            cur += 1
        else:
            if cur * 10 >= min_ms:
                runs.append(cur * 0.01)
            cur = 0
    if cur * 10 >= min_ms:
        runs.append(cur * 0.01)
    return runs


# --------------------------------------------------------------------------- #
# Pitch                                                                        #
# --------------------------------------------------------------------------- #

def _fold_octaves(f0):
    """Correct octave errors against the utterance's median F0.

    Autocorrelation reliably latches onto 2x/0.5x on breathy or high-pitched
    frames.  Those are tracker artefacts, not audible pitch movement, and
    counting them makes a natural voice look like it is shivering.
    """
    v = np.array(f0, dtype=np.float64)
    finite = v[np.isfinite(v) & (v > 0)]
    if len(finite) < 5:
        return v
    med = float(np.median(finite))
    if med <= 0:
        return v
    for i in range(len(v)):
        if not np.isfinite(v[i]) or v[i] <= 0:
            continue
        while v[i] > med * 1.45:
            v[i] *= 0.5
        while v[i] < med * 0.68:
            v[i] *= 2.0
    return v


def f0_track(x, fs, fmin=70.0, fmax=350.0, periodicity=0.4, rms_gate=0.03):
    """Frame-wise F0 (Hz) via autocorrelation; NaN where unvoiced.

    The RMS gate is set above breath-noise level so that deliberate breath
    intakes are not mistaken for voiced speech (which would report bogus
    pitch jumps).
    """
    return _track(x, fs, fmin, fmax, periodicity, rms_gate)[0]


def _track(x, fs, fmin=70.0, fmax=350.0, periodicity=0.4, rms_gate=0.03):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    win = int(fs * 0.04)
    hop = int(fs * 0.01)
    lags = np.arange(int(fs / fmax), int(fs / fmin))
    out = np.full(max(0, (len(x) - win) // hop), np.nan)
    for i, s in enumerate(range(0, len(x) - win, hop)):
        fr = x[s:s + win]
        if len(fr) < win:
            break
        if np.sqrt(np.mean(fr ** 2)) < rms_gate:
            continue
        fr = fr * np.hanning(win)
        ac = np.correlate(fr, fr, "full")[win - 1:]
        ac = ac / (ac[0] + 1e-12)
        L = lags[int(np.argmax(ac[lags]))]
        if ac[L] > periodicity:
            out[i] = fs / L
    return _fold_octaves(out), hop


def pitch_stats(x, fs):
    f0, hop = _track(x, fs)
    v = f0[~np.isnan(f0)]
    if len(v) < 8:
        return {"f0_median": 0.0, "f0_p05": 0.0, "f0_p95": 0.0,
                "jitter_median": 0.0, "jitter_p95": 0.0, "voiced": int(len(v))}
    # Jitter is measured only between TEMPORALLY ADJACENT voiced frames.
    # Diffing across an unvoiced gap (e.g. a breath intake) would compare two
    # unrelated pitches and invent a jump that is not in the audio.
    voiced = ~np.isnan(f0)
    pairs = voiced[:-1] & voiced[1:]
    d = np.abs(f0[1:][pairs] - f0[:-1][pairs]) if pairs.any() else np.array([0.0])
    return {
        "f0_median": float(np.median(v)),
        "f0_p05": float(np.percentile(v, 5)),
        "f0_p95": float(np.percentile(v, 95)),
        "jitter_median": float(np.median(d)),
        "jitter_p95": float(np.percentile(d, 95)),
        "voiced": int(len(v)),
    }


# --------------------------------------------------------------------------- #
# Spectral balance / clarity                                                   #
# --------------------------------------------------------------------------- #

def bands(x, fs):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)

    def b(lo, hi):
        m = (f >= lo) & (f < hi)
        return 10 * np.log10(np.mean(X[m]) + 1e-20)

    return {
        "low": b(120, 500),
        "body": b(500, 1500),
        "mid": b(1500, 4000),
        "high": b(4000, 9000),
        "air": b(9000, 14000),
    }


def clarity(x, fs):
    """Consonant energy relative to vowel body — proxy for phonetic clarity."""
    b = bands(x, fs)
    b["clarity"] = (b["mid"] + b["high"] + b["air"]) / 3.0 - b["body"]
    b["zcr"] = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) * fs
    return b


def loudness(x, fs):
    ed = envelope_db(x, fs)
    active = ed[ed > -50.0]
    return {
        "rms_db": db(np.sqrt(np.mean(x ** 2))),
        "active_rms_db": float(np.mean(active)) if len(active) else db(0.0),
        "loudness_range_db": float(np.percentile(active, 95) - np.percentile(active, 10))
        if len(active) > 4 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Modulation / naturalness                                                     #
# --------------------------------------------------------------------------- #

def modulation(x, fs, lo=0.5, hi=60.0):
    """Energy of amplitude modulation in a band (shimmer / warble indicator)."""
    h = int(fs * 0.01)
    n = len(x) // h
    if n < 16:
        return 0.0
    e = np.array([np.sqrt(np.mean(x[i * h:(i + 1) * h] ** 2)) for i in range(n)])
    m = e.mean()
    if m <= 0:
        return -120.0
    e = np.where(e > m * 0.25, e, np.nan)
    e = e - np.nanmean(e)
    e = np.nan_to_num(e)
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    f = np.fft.rfftfreq(len(e), 0.01)
    tot = np.sum(E) + 1e-20
    return float(10 * np.log10(np.sum(E[(f >= lo) & (f <= hi)]) / tot + 1e-20))


def energy_dips(x, fs, drop_db=12.0, win_ms=20.0, voiced_db=-32.0):
    """Count energy holes that occur INSIDE continuous voicing.

    Zero-padded OLA frames used to punch amplitude holes into the middle of
    speech.  A naive 'any deep dip' test also flags every phrase boundary and
    plosive closure, so a dip only counts when both neighbours are still
    voiced — that is the signature of a processing artefact rather than a
    natural consonant or pause.
    """
    ed = envelope_db(x, fs, win_ms)
    dips = 0
    for i in range(1, len(ed) - 1):
        nb = max(ed[max(i - 2, 0)], ed[i - 1], ed[i + 1], ed[min(i + 2, len(ed) - 1)])
        if ed[i] < nb - drop_db and ed[i - 1] > voiced_db and ed[i + 1] > voiced_db:
            dips += 1
    return dips


def measure(path):
    """Full metric bundle for one rendered file."""
    x, fs = load(path)
    m = {}
    m.update(integrity(x, fs))
    m.update(pitch_stats(x, fs))
    m.update(clarity(x, fs))
    m.update(loudness(x, fs))
    m["mod_hf"] = modulation(x, fs)
    m["energy_dips"] = energy_dips(x, fs)
    m["max_silence"] = max(silence_runs(x, fs)) if silence_runs(x, fs) else 0.0
    m["sample_rate"] = int(fs)
    return m

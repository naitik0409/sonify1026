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


# --------------------------------------------------------------------------- #
# Emotion-differentiation features                                            #
#                                                                             #
# Grounded in the standard acoustic-emotion literature (Eyben et al. 2010,    #
# Pereira 1998, Devillers 2003): the features that reliably separate emotion  #
# states are prosodic — F0 mean/range, intensity, articulation rate — with    #
# spectral balance carrying valence.  All are computed from the waveform only #
# so the benchmark is objective and reproducible.                             #
# --------------------------------------------------------------------------- #

def spectral_centroid(x, fs):
    """Energy-weighted mean frequency (Hz) — spectral 'brightness'."""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    tot = np.sum(X) + 1e-20
    return float(np.sum(f * X) / tot)


def _frame_signal(x, fs, win_s=0.025, hop_s=0.010):
    win = max(16, int(fs * win_s))
    hop = max(1, int(fs * hop_s))
    if len(x) < win:
        return np.zeros((0, win)), win
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.asarray(x)[idx] * np.hanning(win), win


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mfcc(x, fs, n_mfcc=13, n_mels=26, fmin=20.0, fmax=None):
    """Frame-wise MFCCs (n_frames, n_mfcc) — numpy/scipy only, no librosa.

    Power spectrum -> triangular mel filterbank -> log -> DCT-II.  These are
    the standard 'voice colour' features used in emotion and speaker research;
    their means over an utterance characterise timbre independent of loudness
    once c0 (the energy coefficient) is dropped.
    """
    fmax = fmax if fmax is not None else fs / 2.0
    fr, win = _frame_signal(x, fs)
    if len(fr) == 0:
        return np.zeros((0, n_mfcc))
    n_fft = 1 << (int(win) - 1).bit_length()
    spec = np.abs(np.fft.rfft(fr, n_fft)) ** 2

    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * mel_pts / fs).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / float(b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / float(c - b)
    mel_e = np.maximum(spec @ fb.T, 1e-12)
    log_e = np.log(mel_e)
    # DCT-II (orthonormal) — the scipy definition, implemented directly so the
    # test suite keeps its current dependency set.
    n = n_mels
    basis = np.cos(np.pi * np.arange(n_mfcc)[:, None] *
                   (2 * np.arange(n)[None, :] + 1) / (2.0 * n))
    basis[0] *= np.sqrt(1.0 / n)
    basis[1:] *= np.sqrt(2.0 / n)
    return log_e @ basis.T


def cepstral_mean(x, fs, n_mfcc=13):
    """Utterance-level mean MFCC vector — a compact timbre/identity descriptor."""
    c = mfcc(x, fs, n_mfcc=n_mfcc)
    if len(c) == 0:
        return np.zeros(n_mfcc)
    return c.mean(axis=0)


def mcd(c1, c2):
    """Mel cepstral distortion (dB) between two mean-MFCC vectors.

    Classic voice-conversion speaker-individuality measure:
        MCD = (10/ln10) * sqrt(2 * sum_m (c1_m - c2_m)^2)
    c0 is excluded by the caller (it mostly tracks energy, not identity).
    Identical timbres -> 0; different speakers -> typically 4-8 dB.
    """
    a = np.asarray(c1, dtype=np.float64).ravel()
    b = np.asarray(c2, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    d = a[1:n] - b[1:n]          # drop c0 (energy) — identity lives above it
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(d ** 2)))


def syllable_nuclei(x, fs):
    """Count syllable nuclei via energy-envelope peak picking.

    The standard cheap articulation-rate proxy (de Jong & Wempe): nuclei are
    local maxima of the dB envelope above a relative threshold, at least
    200 ms apart (physiological ceiling on syllable rate).
    """
    ed = envelope_db(x, fs, win_ms=10.0)
    if len(ed) < 8:
        return 0
    from scipy.signal import find_peaks
    base = float(np.median(ed))
    top = float(np.percentile(ed, 95))
    thr = base + 0.35 * max(top - base, 1.0)
    peaks, _ = find_peaks(ed, height=thr, distance=max(1, int(0.20 / 0.01)))
    return int(len(peaks))


def emotion_features(x, fs):
    """Acoustic feature bundle for emotion-differentiation benchmarking.

    One flat dict of scalars so rows from different renders can be z-scored
    and compared directly (see tests/emotion_benchmark.py).
    """
    p = pitch_stats(x, fs)
    l = loudness(x, fs)
    c = clarity(x, fs)
    dur = len(x) / fs
    nuclei = syllable_nuclei(x, fs)
    return {
        "f0_median": p["f0_median"],
        "f0_span": p["f0_p95"] - p["f0_p05"],
        "rms_db": l["rms_db"],
        "dyn_db": l["loudness_range_db"],
        "rate_syll": nuclei / dur if dur > 0 else 0.0,
        "nuclei": float(nuclei),
        "duration": dur,
        # pace = 1/duration.  With identical text per language this is the
        # robust articulation-rate cue: the syllable-nuclei detector is noisy
        # on Hindi retroflexes, while duration cleanly tracks both the emotion
        # rate offset and the configured pause multiplier.
        "pace": 1.0 / dur if dur > 0 else 0.0,
        "clarity": c["clarity"],
        "centroid_hz": spectral_centroid(x, fs),
    }


# Features that carry emotion information for the separation matrix.
# Excluded on purpose:
#   rms_db   — the mastering stage normalises every render to target_rms_db,
#              so absolute level cannot differ between cells by design;
#   rate_syll — noisy syllable counting; `pace` (1/duration) is the reliable
#              rate cue when the text is held constant.
EMOTION_FEATURE_KEYS = (
    "f0_median", "f0_span", "dyn_db", "pace", "clarity", "centroid_hz",
)

# Acoustic correlates of arousal (high -> low): F0 level and tempo are the
# two dimensions the emotion presets control that survive normalisation —
# exactly the classic arousal pair from the emotion literature (Eyben 2010,
# Pereira 1998).  F0 span / loudness dynamics vary too but are contaminated
# by phrase-level effects, so they inform separation, not the rank test.
AROUSAL_FEATURE_KEYS = ("f0_median", "pace")


def identity_features(x, fs, n_mfcc=13):
    """Speaker-identity descriptor: pitch statistics + mean timbre vector."""
    p = pitch_stats(x, fs)
    f0 = f0_track(x, fs)
    v = f0[np.isfinite(f0) & (f0 > 0)]
    return {
        "f0_median": p["f0_median"],
        "f0_iqr": float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 4 else 0.0,
        "mfcc_mean": cepstral_mean(x, fs, n_mfcc=n_mfcc),
    }

"""
Sonify 1026 Voice Engine
A human-like TTS performance layer over Microsoft Edge TTS: the neural voice
and audio come from Edge TTS; Sonify does not ship its own speech model, it
directs and post-processes that voice for emotion, accent and prosody.
Uses edge-tts as the primary backend with pyttsx3 as an offline fallback
(a different, lower-quality OS voice, not the same sound).
Includes advanced DSP: FFT EQ, breath sounds, room air, glue compression.
"""

import os
import re
import random
import asyncio
import tempfile
import numpy as np
import yaml
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sonify1026")

SR_OUT = 48000


class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    WHISPER = "whisper"
    FEARFUL = "fearful"
    WARM = "warm"
    GENTLE = "gentle"
    THOUGHTFUL = "thoughtful"
    PLAYFUL = "playful"
    TIRED = "tired"
    PROUD = "proud"
    SURPRISED = "surprised"
    REASSURING = "reassuring"
    SERIOUS = "serious"


class Language(Enum):
    EN_IN = "en-IN"
    HI_IN = "hi-IN"
    EN_US = "en-US"


@dataclass
class ProsodyParams:
    pitch_shift: float = 0.0
    rate_scale: float = 1.0
    energy: float = 1.0
    pause_factor: float = 1.0


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration: float
    emotion: Emotion
    language: Language
    text: str


# --------------------------------------------------------------------------- #
# Persona & voices                                                             #
# --------------------------------------------------------------------------- #

class Persona:
    """Sonify 1026: a warm, expressive, articulate Indian female presence."""
    name = "Sonify 1026"
    base_pitch = 0
    highpass_hz = 85.0
    warm_hz, warm_gain = 180.0, 0.4
    pres_hz, pres_gain = 3400.0, 0.4
    target_rms_db = -20.5
    peak_ceiling_db = -1.0


VOICES = {
    "neerja": dict(id="en-IN-NeerjaNeural", pitch=0, lo=-20, hi=25),
    # Swara: Slightly lowered pitch (~272 Hz) for both English & Hindi
    "swara": dict(id="hi-IN-SwaraNeural", pitch=0, lo=-20, hi=25),
    "aria": dict(id="en-US-AriaNeural", pitch=0, lo=-20, hi=25),
}

ACCENTS = {
    "in": dict(voice="neerja", rate=-2, label="Indian English (Neerja)"),
    "us": dict(voice="aria", rate=0, label="General American"),
}
DEFAULT_ACCENT = "in"

# Voice selection mode.  "single" uses one voice for every language so the
# speaker identity (and accent) never changes mid-script; "dual" keeps the
# legacy English-vs-Devanagari split.  The primary must be a voice the
# backend can drive across scripts (the hi-IN voices are bilingual).
VOICE_MODE = {"mode": "dual", "primary": "swara"}

DEV = re.compile(r"[\u0900-\u097F]")

# Keep English in one neural take so commas/questions carry human intonation.
MAX_UTTERANCE_CHARS = 480


def _is_hindi(t):
    return bool(DEV.search(t))


def _pick_voice(text, accent):
    if VOICE_MODE.get("mode", "dual") == "single":
        return VOICE_MODE.get("primary", "swara")
    if _is_hindi(text):
        return "swara"
    return ACCENTS.get(accent, ACCENTS[DEFAULT_ACCENT])["voice"]


EMOTIONS = {
    "neutral": dict(pitch=0, rate="-2%", volume="+0%", pause=1.00, bright=0.0, warm=0.15, dark=None, comp=1.35, air=-48),
    "warm": dict(pitch=1, rate="-4%", volume="-1%", pause=1.06, bright=0.15, warm=0.5, dark=None, comp=1.35, air=-47),
    "happy": dict(pitch=2, rate="+2%", volume="+2%", pause=0.96, bright=0.35, warm=0.1, dark=None, comp=1.4, air=-48),
    "calm": dict(pitch=-1, rate="-6%", volume="-2%", pause=1.12, bright=-0.2, warm=0.4, dark=None, comp=1.25, air=-47),
    "gentle": dict(pitch=-1, rate="-5%", volume="-4%", pause=1.10, bright=-0.15, warm=0.45, dark=None, comp=1.25, air=-47),
    "sad": dict(pitch=-3, rate="-8%", volume="-5%", pause=1.22, bright=-0.6, warm=0.6, dark=7800.0, comp=1.2, air=-47),
    "excited": dict(pitch=3, rate="+5%", volume="+3%", pause=0.90, bright=0.5, warm=0.0, dark=None, comp=1.45, air=-49),
    "whisper": dict(pitch=-2, rate="-6%", volume="-12%", pause=1.18, bright=0.2, warm=0.25, dark=None, comp=1.2, air=-54),
    "angry": dict(pitch=1, rate="+2%", volume="+4%", pause=0.97, bright=0.35, warm=0.2, dark=None, comp=1.55, air=-49),
    "fearful": dict(pitch=2, rate="+4%", volume="+1%", pause=0.94, bright=0.2, warm=-0.1, dark=None, comp=1.4, air=-48),
    "thoughtful": dict(pitch=-1, rate="-5%", volume="-2%", pause=1.14, bright=-0.1, warm=0.4, dark=None, comp=1.25, air=-47),
    "playful": dict(pitch=2, rate="+3%", volume="+2%", pause=0.94, bright=0.4, warm=0.0, dark=None, comp=1.4, air=-48),
    "tired": dict(pitch=-2, rate="-8%", volume="-4%", pause=1.18, bright=-0.5, warm=0.5, dark=8000.0, comp=1.2, air=-46),
    "proud": dict(pitch=1, rate="-1%", volume="+2%", pause=1.04, bright=0.25, warm=0.35, dark=None, comp=1.4, air=-48),
    "surprised": dict(pitch=4, rate="+4%", volume="+3%", pause=0.92, bright=0.45, warm=-0.1, dark=None, comp=1.45, air=-48),
    "reassuring": dict(pitch=0, rate="-3%", volume="-1%", pause=1.10, bright=0.1, warm=0.55, dark=None, comp=1.3, air=-47),
    "serious": dict(pitch=-1, rate="-2%", volume="+1%", pause=1.06, bright=0.1, warm=0.25, dark=None, comp=1.4, air=-48),
}

DEFAULT_EMOTION = "neutral"
DEFAULT_REGISTER = "normal"
SENTENCE_PAUSE_MS = 240
REGISTERS = {
    "normal": dict(rate=0, pitch=0, pause=1.00, vol=0, bright=0.0),
    "formal": dict(rate=-3, pitch=-1, pause=1.12, vol=-1, bright=-0.3),
    "casual": dict(rate=5, pitch=1, pause=0.88, vol=1, bright=0.3),
}


# --------------------------------------------------------------------------- #
# FFT biquad EQ                                                                #
# --------------------------------------------------------------------------- #

def _bj_mag(b, a, n, fs):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    w = 2.0 * np.pi * f / fs
    e1, e2 = np.exp(-1j * w), np.exp(-2j * w)
    num = np.abs(b[0] + b[1] * e1 + b[2] * e2)
    den = np.abs(a[0] + a[1] * e1 + a[2] * e2)
    return num / den


def _peaking(fs, f0, gain_db, q):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b0, b1, b2 = 1.0 + al * A, -2.0 * c, 1.0 - al * A
    a0, a1, a2 = 1.0 + al / A, -2.0 * c, 1.0 - al / A
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _hp(fs, fc, q=0.707):
    w0 = 2.0 * np.pi * fc / fs
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0]
    a0 = 1.0 + al
    b = [x / a0 for x in b]
    a = [1.0, (-2.0 * c) / a0, (1.0 - al) / a0]
    return b, a


def _lp(fs, fc, q=0.707):
    w0 = 2.0 * np.pi * fc / fs
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = [(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0]
    a0 = 1.0 + al
    b = [x / a0 for x in b]
    a = [1.0, (-2.0 * c) / a0, (1.0 - al) / a0]
    return b, a


def _eq(x, fs, sections):
    """
    Phase-PRESERVING biquad EQ.

    The previous implementation multiplied the FFT by a magnitude-only
    response, which discarded all phase information.  That smears
    transients and produces an audible 'shivering'/muffled artefact.
    Filtering each biquad in the time domain with sosfilt keeps the
    waveform intact, so consonants stay crisp.
    """
    if not sections:
        return x
    from scipy.signal import sosfilt

    sos = []
    for s in sections:
        if s[0] == "pk":
            b, a = _peaking(fs, s[1], s[2], s[3])
        elif s[0] == "hp":
            b, a = _hp(fs, s[1])
        elif s[0] == "lp":
            b, a = _lp(fs, s[1])
        else:
            continue
        sos.append([b[0] / a[0], b[1] / a[0], b[2] / a[0],
                    1.0, a[1] / a[0], a[2] / a[0]])
    if not sos:
        return x

    y = np.asarray(x, dtype=np.float64)
    # Short reflect padding keeps the filter state from ringing at the edges
    pad = min(len(y), int(fs * 0.005))
    if pad > 1:
        y = np.concatenate([y[pad:0:-1], y, y[-2:-pad - 2:-1]])
    y = sosfilt(np.asarray(sos, dtype=np.float64), y)
    if pad > 1:
        y = y[pad:len(y) - pad]
    return y[:len(x)]


# --------------------------------------------------------------------------- #
# Punctuation-aware segmenting                                                 #
# --------------------------------------------------------------------------- #

_EM_RE = re.compile(r"\*\*(.+?)\*\*")

_TAG_RE = re.compile(r"\[([A-Za-z]+)\]")


def _split_emotion_tags(text, tag_map):
    """Split text at inline emotion tags like [cheerful] or [softly].

    Returns [(emotion_key_or_None, overrides_dict, span_text)].  A tag
    switches the emotion for everything after it until the next known tag.
    A tag value may be a plain emotion name or a dict with `emotion` plus
    per-tag overrides (pitch_hz, rate_pct, volume_db, bright_db, warm_db).
    Only tags present in the config map cause a split — unknown bracket text
    (e.g. "[sic]") is left untouched so it is never silently eaten.
    """
    OVERRIDE_KEYS = ("pitch_hz", "rate_pct", "volume_db", "bright_db", "warm_db")
    spans = []
    pos = 0
    cur = None
    cur_ovr = {}
    for m in _TAG_RE.finditer(text):
        raw = (tag_map or {}).get(m.group(1).lower())
        if isinstance(raw, dict):
            mapped, ovr = raw.get("emotion"), {k: raw[k] for k in OVERRIDE_KEYS if k in raw}
        else:
            mapped, ovr = raw, {}
        if mapped not in EMOTIONS:
            # Unknown bracket text is left untouched (it may be intentional),
            # but warn — a typo'd tag would otherwise be spoken aloud.
            logger.warning("Unknown emotion tag [%s] left in text; "
                           "add it to emotion_tags to use it.", m.group(1))
            continue
        if m.start() > pos:
            spans.append((cur, cur_ovr, text[pos:m.start()]))
        cur, cur_ovr = mapped, ovr
        pos = m.end()
    if pos < len(text):
        spans.append((cur, cur_ovr, text[pos:]))
    if not spans:
        spans.append((None, {}, text))
    return spans

_ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "st", "vs", "etc", "e.g", "i.e",
    "no", "fig", "approx", "inc", "ltd", "dept", "vol", "pp", "jr", "co",
    "ca", "cf", "ref", "est", "col", "gen", "sgt", "lt", "rev", "hon",
    "mt", "ft", "al", "viz", "ie", "eg", "ed", "eds", "min", "max",
    "incl", "excl", "univ",
}

DEV_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')

UNITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
         'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
         'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

ORDINALS = {
    1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
    6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
    11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth',
    15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth', 19: 'nineteenth'
}

def _num_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _num_to_words(-n)
    if n < 20:
        return UNITS[n]
    if n < 100:
        return TENS[n // 10] + ('' if n % 10 == 0 else '-' + UNITS[n % 10])
    if n < 1000:
        return UNITS[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + _num_to_words(n % 100))
    if n < 100000:
        return _num_to_words(n // 1000) + ' thousand' + ('' if n % 1000 == 0 else ' ' + _num_to_words(n % 1000))
    if n < 10000000:
        return _num_to_words(n // 100000) + ' lakh' + ('' if n % 100000 == 0 else ' ' + _num_to_words(n % 100000))
    if n < 1000000000:
        return _num_to_words(n // 10000000) + ' crore' + ('' if n % 10000000 == 0 else ' ' + _num_to_words(n % 10000000))
    return ' '.join(UNITS[int(d)] for d in str(n))


def _year_to_words(n: int) -> str:
    if n % 100 == 0:
        return _num_to_words(n)
    return _num_to_words(n // 100) + ' ' + _num_to_words(n % 100)


def _is_phone_digits(val: str) -> bool:
    if not val.isdigit():
        return False
    n = len(val)
    if n == 10 and val[0] in "6789":
        return True
    if n == 11 and val.startswith("0"):
        return True
    if n == 12 and val.startswith("91"):
        return True
    return n >= 10

def _num_to_ordinal(n: int) -> str:
    if n in ORDINALS:
        return ORDINALS[n]
    if n < 100:
        ten = n // 10
        unit = n % 10
        if unit == 0:
            tens_ord = {'twenty': 'twentieth', 'thirty': 'thirtieth', 'forty': 'fortieth',
                        'fifty': 'fiftieth', 'sixty': 'sixtieth', 'seventy': 'seventieth',
                        'eighty': 'eightieth', 'ninety': 'ninetieth'}
            return tens_ord.get(TENS[ten], TENS[ten] + 'th')
        return TENS[ten] + '-' + ORDINALS.get(unit, _num_to_words(unit) + 'th')
    return _num_to_words(n) + 'th'

def _replace_ordinal_match(m):
    num_part = int(m.group(1).translate(DEV_DIGITS))
    return _num_to_ordinal(num_part)

def _replace_number_match(m):
    val = m.group(0).translate(DEV_DIGITS)
    if '.' in val:
        parts = val.split('.')
        left = _num_to_words(int(parts[0])) if parts[0] else 'zero'
        right = ' '.join(UNITS[int(d)] if d.isdigit() else d for d in parts[1])
        return f'{left} point {right}'
    if len(val) > 1 and val.startswith('0'):
        return ' '.join(UNITS[int(d)] for d in val)
    if _is_phone_digits(val):
        return ' '.join(UNITS[int(d)] for d in val)
    n = int(val)
    if len(val) == 4 and 1900 <= n <= 2099:
        return _year_to_words(n)
    return _num_to_words(n)

def convert_numbers_to_english(text: str) -> str:
    text = text.translate(DEV_DIGITS)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', _replace_ordinal_match, text, flags=re.IGNORECASE)
    return re.sub(r'\b\d+(?:\.\d+)?\b', _replace_number_match, text)

SALUTATIONS = [
    r'\bhello\s+sir\b', r'\bhello\s+mam\b', r'\bhello\s+ma\'am\b',
    r'\bhi\s+sir\b', r'\bhi\s+mam\b', r'\bhi\s+ma\'am\b',
    r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bnamaste\b', r'\bokay\b', r'\bok\b',
    r'\bhaan\b', r'\bwell\b', r'\bso\b', r'\bactually\b', r'\blisten\b',
    r'\bachha\b', r'\baccha\b', r'\btheek\s+hai\b',
]
SALUTATION_RE = re.compile(r'^(' + '|'.join(SALUTATIONS) + r')\s+([^\,\.\!\?])', re.IGNORECASE)

_CONTRACTIONS = [
    (re.compile(r"\bI am\b"), "I'm"),
    (re.compile(r"\bI've\b"), "I've"),
    (re.compile(r"\bI will\b"), "I'll"),
    (re.compile(r"\bI would\b"), "I'd"),
    (re.compile(r"\byou are\b"), "you're"),
    (re.compile(r"\byou're\b"), "you're"),
    (re.compile(r"\bwe are\b"), "we're"),
    (re.compile(r"\bthey are\b"), "they're"),
    (re.compile(r"\bit is\b"), "it's"),
    (re.compile(r"\bIt is\b"), "It's"),
    (re.compile(r"\bthat is\b"), "that's"),
    (re.compile(r"\bThat is\b"), "That's"),
    (re.compile(r"\bwhat is\b"), "what's"),
    (re.compile(r"\bWhat is\b"), "What's"),
    (re.compile(r"\bwhere is\b"), "where's"),
    (re.compile(r"\bWhere is\b"), "Where's"),
    (re.compile(r"\bdo not\b"), "don't"),
    (re.compile(r"\bDo not\b"), "Don't"),
    (re.compile(r"\bdoes not\b"), "doesn't"),
    (re.compile(r"\bcannot\b"), "can't"),
    (re.compile(r"\bwill not\b"), "won't"),
    (re.compile(r"\bis not\b"), "isn't"),
    (re.compile(r"\bare not\b"), "aren't"),
    (re.compile(r"\bwould not\b"), "wouldn't"),
    (re.compile(r"\bcould not\b"), "couldn't"),
    (re.compile(r"\bshould not\b"), "shouldn't"),
    (re.compile(r"\blet us\b"), "let's"),
    (re.compile(r"\bLet us\b"), "Let's"),
    (re.compile(r"\bI have\b"), "I've"),
]


def _apply_contractions(text: str) -> str:
    for pat, repl in _CONTRACTIONS:
        text = pat.sub(repl, text)
    return text


def _preprocess(text, contractions=True, verbatim=False):
    t = text.replace("\r\n", "\n")
    if not verbatim:
        t = t.replace("₹", " rupees ")
        t = re.sub(r"\bRs\.?\s*", " rupees ", t, flags=re.IGNORECASE)
        t = convert_numbers_to_english(t)
    t = re.sub(r"\n+", " ", t)
    if not verbatim:
        t = re.sub(r"\s+(?:—|–|--)\s+", ", ", t)
        t = t.replace("...", ",")
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    if contractions and not verbatim and not _is_hindi(t):
        t = _apply_contractions(t)
    if not verbatim:
        t = SALUTATION_RE.sub(r'\1, \2', t)
    if t and t[-1] not in '.!?…;:,।':
        t += '.'
    return t


def _kind_of(ch):
    return {"?": "question", "!": "exclaim", "…": "ellipsis",
            ";": "clause", ":": "clause", ",": "clause", "।": "sentence"}.get(ch, "sentence")


def _split_kinds(text):
    """
    Split into pieces labelled by their own terminating punctuation.

    A piece's kind is decided ONLY by the punctuation that ends it, so a
    comma-separated fragment never inherits the 'clause' label of the
    previous fragment.
    """
    pieces = []
    last = 0
    for m in re.finditer(r"(?<=[.!?…;:,।])\s+|(?<=[।])", text):
        end = m.start()
        punct = text[end - 1] if end > 0 else ""
        if punct == ".":
            before = text[:end]
            tok = re.split(r"[\s\"'“”‘’(\[]+", before.rstrip())[-1]
            core = tok.lstrip("\"'“‘([").rstrip(".").lower()
            if core in _ABBREV or (len(core) == 1 and core.isalpha()) or core == "":
                continue
        raw = text[last:end].strip()
        if raw:
            pieces.append((raw, _kind_of(punct)))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        stripped = tail.rstrip()
        k = _kind_of(stripped[-1]) if stripped and stripped[-1] in "?!…;:,।" else "sentence"
        pieces.append((tail, k))
    return pieces


def _rekind(piece):
    """Re-derive a piece's kind from its own final punctuation."""
    txt = piece.rstrip()
    if txt and txt[-1] in "?!…;:,।":
        return _kind_of(txt[-1])
    return "sentence"


def _pack_utterances(text, max_chars=MAX_UTTERANCE_CHARS, config=None):
    """
    Group text into utterances for the neural voice.

    Every take must END on sentence-final punctuation and must contain
    whole words.  Cutting a take mid-clause makes the neural voice re-read
    the seam, which is heard as a word repeated at the same time, and an
    abrupt onset/offset that reads as 'shivering'.
    """
    cfg = (config or {}).get("humanize", {})
    soft_max = float(cfg.get("soft_max_utterance_chars", max_chars * 1.6))
    soft_max = max(float(max_chars), soft_max)

    # Phase A — clause-level takes.  When enabled, a take may also close on a
    # clause boundary (comma / semicolon / colon) instead of only a sentence
    # end, so long multi-clause sentences are read as several shorter phrases
    # with their own intonation rather than one flat run.  Every piece from
    # `_split_kinds` already ends on punctuation, so the longest take that
    # fits the cap always closes on a real boundary.
    ct = cfg.get("clause_takes", {}) or {}
    clause_takes = bool(ct.get("enabled", False))
    if clause_takes:
        soft_max = max(float(max_chars), float(ct.get("max_utterance_chars", soft_max)))

    pieces = _split_kinds(text)
    if not pieces:
        return []

    # Absorb clause fragments into the sentence that contains them.  A take
    # must never END mid-clause.  When the next fragment would overflow the
    # limit we backtrack to the last complete sentence in the buffer and
    # carry the remainder into a fresh take.
    packed = []
    buf = []          # list of (text, ends_sentence)
    buf_len = 0

    def flush(upto):
        nonlocal buf, buf_len
        take = " ".join(t for t, _ in buf[:upto])
        packed.append((take, _rekind(take)))
        rest = buf[upto:]
        buf = rest
        buf_len = sum(len(t) + 1 for t, _ in rest)

    for raw, kind in pieces:
        ends = raw.rstrip().endswith((".", "!", "?", "…", "।"))
        extra = len(raw) + (1 if buf else 0)
        if buf and buf_len + extra > soft_max:
            if clause_takes:
                # Close the take on the last (clause or sentence) boundary —
                # every buffered piece ends on punctuation, so this is always
                # a valid prosodic break.
                cut = len(buf)
            else:
                # Find the last complete sentence to close the take at.
                cut = len(buf)
                for i in range(len(buf) - 1, -1, -1):
                    if buf[i][1]:
                        cut = i + 1
                        break
                if cut == 0:
                    cut = 1          # single oversized fragment: take it as-is
            flush(cut)
        buf.append((raw, ends))
        buf_len += len(raw) + (1 if len(buf) > 1 else 0)
    if buf:
        take = " ".join(t for t, _ in buf)
        packed.append((take, _rekind(take)))
    return packed


def _gap_ms(kind, em, rng, config=None):
    if kind is None:
        return 0.0
    punct_cfg = (config or {}).get("punctuation", {})
    # Per-kind base pause; falls back to sentence_pause_ms x kind multiplier.
    kind_cfg = punct_cfg.get(kind, {}) or {}
    if isinstance(kind_cfg.get("pause_ms"), (int, float)):
        base = float(kind_cfg["pause_ms"])
    else:
        base_ms = float(punct_cfg.get("sentence_pause_ms", SENTENCE_PAUSE_MS))
        mult = float((punct_cfg.get("kind_mult", {}) or {}).get(kind, 1.0))
        base = base_ms * mult
    jitter = float(punct_cfg.get("pause_jitter", 0.05))
    jitter = float(np.clip(jitter, 0.0, 0.9))
    gap = base * em["pause"] * rng.uniform(1.0 - jitter, 1.0 + jitter)
    # Slow cognitive pause — "thinking of the words" stall
    cog = punct_cfg.get("cognitive_pause", {}) or {}
    if rng.random() < float(cog.get("probability", 0.0)):
        gap *= rng.uniform(float(cog.get("min_factor", 1.6)),
                           float(cog.get("max_factor", 2.3)))
    # Thought gap — a deliberate beat at a thought boundary, as if considering
    # the next thought rather than just pausing for punctuation.
    tg = (config or {}).get("thought_gap", {}) or {}
    if tg.get("enabled", False):
        kinds = tg.get("kinds", ["sentence", "clause"])
        if (kind in kinds
                and rng.random() < float(tg.get("probability", 0.0))):
            gap *= rng.uniform(float(tg.get("min_factor", 1.3)),
                               float(tg.get("max_factor", 1.9)))
    return gap


def _segment(text, emotion, persona, accent_rate, rng, register="normal",
             variation=None, config=None, verbatim=False):
    """
    Segment text into prosody-labelled chunks.  ALL adjustment rules
    (sentence-type, aside, dynamic intensity) are read from config —
    nothing is hardcoded in this function.

    Inline emotion tags like [cheerful] or [softly] switch the emotion for
    the text that follows them (see `emotion_tags` in config).
    """
    emotion = emotion.value if isinstance(emotion, Emotion) else emotion
    reg = REGISTERS.get(register, REGISTERS["normal"])
    var = variation or {}
    cfg = config or {}
    type_rules = cfg.get("sentence_type_rules", {})
    human = cfg.get("humanize", {})
    max_chars = int(human.get("max_utterance_chars", MAX_UTTERANCE_CHARS))
    use_contractions = human.get("contractions", True) and register != "formal"

    speed = float(cfg.get("_speed", 1.0))
    pitch_off = float(cfg.get("_pitch", 0.0))
    vol_scale = float(cfg.get("_volume", 1.0))
    speed_rate = int(round((speed - 1.0) * 100.0))
    vol_db = 0 if vol_scale >= 0.999 else int(round(20.0 * np.log10(max(vol_scale, 1e-3))))

    norm     = _preprocess(text, contractions=use_contractions, verbatim=verbatim)
    tag_map  = cfg.get("emotion_tags", {})
    segments = []
    for span_emo, span_ovr, span_text in _split_emotion_tags(norm, tag_map):
        if not span_text.strip():
            continue
        seg_emo = span_emo or emotion
        em = EMOTIONS.get(seg_emo, EMOTIONS[DEFAULT_EMOTION])
        ovr = span_ovr or {}
        for raw, kind in _pack_utterances(span_text, max_chars=max_chars, config=cfg):
            clean = _EM_RE.sub(lambda m: m.group(1), raw).strip()
            if not clean:
                continue
            emph_words = []
            for m in _EM_RE.finditer(raw):
                emph_words.extend(re.findall(r"[\w'']+", m.group(1)))

            # ---- Config-driven sentence-type adjustments (no hardcoding) ----
            is_aside = raw.lstrip().startswith("(")
            if is_aside:
                clean = clean.strip("()").strip()
                aside_rule = type_rules.get("aside", {})
                p_adj = aside_rule.get("pitch",  -2)
                r_adj = aside_rule.get("rate",   -1)
                v_adj = aside_rule.get("volume", -2)
            else:
                rule  = type_rules.get(kind or "sentence", {})
                p_adj = rule.get("pitch",  0)
                r_adj = rule.get("rate",   0)
                v_adj = rule.get("volume", 0)

            # ---- Dynamic text analysis (intensity words + sentence length) ----
            dyn    = _analyze_text_dynamics(clean, cfg)
            p_adj += dyn.get("pitch_delta", 0.0) + float(ovr.get("pitch_hz", 0.0))
            r_adj += dyn.get("rate_delta",  0.0) + float(ovr.get("rate_pct", 0.0))
            v_adj += dyn.get("volume_delta", 0.0) + float(ovr.get("volume_db", 0.0))

            pitch_var = var.get("pitch_hz",   0.6)
            rate_var  = var.get("rate_pct",   0.6)
            vol_var   = var.get("volume_db",  0.4)

            char_cfg = cfg.get("character", {})
            p_lo = int(char_cfg.get("pitch_min", -20))
            p_hi = int(char_cfg.get("pitch_max", 25))
            pitch = int(np.clip(persona.base_pitch + em["pitch"] + p_adj
                                + reg["pitch"] + pitch_off
                                + rng.uniform(-pitch_var, pitch_var), p_lo, p_hi))
            # Fast burst — occasionally a quick run of words
            burst = var.get("burst", {}) or {}
            burst_delta = 0.0
            if rng.random() < float(burst.get("probability", 0.0)):
                burst_delta = rng.uniform(float(burst.get("min_pct", 4.0)),
                                          float(burst.get("max_pct", 9.0)))
            rate  = _shift_pct(em["rate"],
                               accent_rate + r_adj + reg["rate"] + speed_rate
                               + burst_delta
                               + rng.uniform(-rate_var, rate_var))
            vol   = int(np.clip(int(em["volume"].rstrip("%")) + reg["vol"] + v_adj
                                + vol_db + rng.uniform(-vol_var, vol_var), -30, 20))
            segments.append(dict(
                text=clean,
                pitch=pitch,
                rate=rate,
                volume=f"{vol:+d}%",
                bright=em["bright"] + reg["bright"] + dyn.get("bright_delta", 0.0)
                        + float(ovr.get("bright_db", 0.0)),
                warm=em["warm"] + dyn.get("warm_delta", 0.0)
                      + float(ovr.get("warm_db", 0.0)),
                pause_ms=_gap_ms(kind, em, rng, cfg) * reg["pause"]
                          * float(dyn.get("pause_scale", 1.0)),
                emph=[w.lower() for w in emph_words],
                key_words=dyn.get("key_words", []),
                low_words=dyn.get("low_words", []),
                breath_scale=dyn.get("breath_scale", 1.0),
                effort=dyn.get("effort", 0.0),
                valence=dyn.get("valence", 0.0),
                emo=seg_emo,
                kind=kind,
            ))
    return segments


def _shift_pct(rate_str, delta):
    v = int(rate_str.rstrip("%"))
    v = int(np.clip(v + delta, -50, 100))
    return f"{v:+d}%"


# --------------------------------------------------------------------------- #
# edge-tts segment synthesis                                                   #
# --------------------------------------------------------------------------- #

# In-process cache of raw backend audio, keyed by the exact synthesis
# parameters.  Off by default (see the `backend_cache` config section): when
# enabled, repeated identical takes reuse the same neural audio, which makes
# output reproducible for a fixed seed and avoids redundant network calls.
_EDGE_AUDIO_CACHE: Dict[tuple, tuple] = {}


async def _edge_one(seg, voice):
    import edge_tts
    com = edge_tts.Communicate(
        seg["text"], voice,
        rate=seg["rate"], volume=seg["volume"], pitch=f"{seg['pitch']:+d}Hz",
        boundary="WordBoundary",
    )
    audio = bytearray()
    words = []
    async for ch in com.stream():
        if ch["type"] == "audio":
            audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            words.append((ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]))
    return bytes(audio), words


async def _edge_all(segments, default_voice, cache_cfg=None):
    cache_cfg = cache_cfg or {}
    use_cache = bool(cache_cfg.get("enabled", False))
    max_entries = int(cache_cfg.get("max_entries", 64) or 64)
    out = []
    for i, seg in enumerate(segments):
        if i:
            await asyncio.sleep(0.08)
        vkey = seg.get("vkey") or default_voice
        voice = VOICES.get(vkey, VOICES["neerja"])["id"]
        key = (seg["text"], voice, seg.get("rate"), seg.get("volume"),
               f"{seg.get('pitch', 0):+d}Hz")
        if use_cache and key in _EDGE_AUDIO_CACHE:
            out.append(_EDGE_AUDIO_CACHE[key])
            continue
        try:
            result = await _edge_one(seg, voice)
        except Exception as e:
            # The backend returns no audio for degenerate input (e.g. a bare
            # ellipsis).  Skip that take instead of failing the whole render.
            logger.warning("edge-tts produced no audio for a segment: %s", e)
            result = (b"", [])
        if use_cache:
            _EDGE_AUDIO_CACHE[key] = result
            while len(_EDGE_AUDIO_CACHE) > max_entries:
                _EDGE_AUDIO_CACHE.pop(next(iter(_EDGE_AUDIO_CACHE)))
        out.append(result)
    return out


def clear_backend_cache():
    """Drop the in-process raw-audio cache (see config `backend_cache`)."""
    _EDGE_AUDIO_CACHE.clear()


def _decode_mp3(data):
    x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


# --------------------------------------------------------------------------- #
# Per-segment processing                                                       #
# --------------------------------------------------------------------------- #

def _trim(x, fs, margin_s=0.08):
    w = max(1, int(fs * 0.010))
    n = (len(x) // w) * w
    if n == 0:
        return x, 0.0
    frames = x[:n].reshape(-1, w)
    fr = np.sqrt((frames ** 2).mean(axis=1) + 1e-20)
    peak = float(np.max(np.abs(x)) + 1e-12)
    thr = max(peak * 10 ** (-48 / 20.0), 10 ** (-58 / 20.0))
    voiced = np.where(fr > thr)[0]
    if len(voiced) == 0:
        return x, 0.0
    a = max(0, voiced[0] * w - int(margin_s * fs))
    b = min(len(x), (voiced[-1] + 1) * w + int(margin_s * fs))
    return x[a:b], a / fs


def _fade(x, fs, ms=5.0):
    n = min(len(x), max(1, int(fs * ms / 1000.0)))
    if len(x) < 2 * n:
        return x
    x[:n] *= np.linspace(0.0, 1.0, n)
    x[-n:] *= np.linspace(1.0, 0.0, n)
    return x


def _compress_gaps(x: np.ndarray, fs: int,
                   config: Optional[Dict] = None) -> np.ndarray:
    """Shorten over-long interior silences to a natural micro-pause.

    The neural backend inserts its own pauses inside long takes, which can
    run far beyond the configured pause range.  Any interior silence longer
    than `max_silence_ms` is shortened to `target_ms` with a short crossfade
    so no click is introduced.  Leading/trailing file silence is untouched.
    Config: 'gap_compress' in voice_config.yaml.
    """
    cfg = (config or {}).get("gap_compress", {})
    if not cfg.get("enabled", False) or len(x) < fs // 2:
        return x
    thr_db  = float(cfg.get("threshold_db", -50.0))
    max_ms  = float(cfg.get("max_silence_ms", 220.0))
    tgt_ms  = float(cfg.get("target_ms", 160.0))
    fade_ms = float(cfg.get("fade_ms", 12.0))
    if tgt_ms >= max_ms or tgt_ms <= 0:
        return x

    thr = 10.0 ** (thr_db / 20.0)
    h = max(1, int(fs * 0.01))
    n = len(x) // h
    if n < 4:
        return x
    env = np.array([np.sqrt(np.mean(x[i * h:(i + 1) * h] ** 2))
                    for i in range(n)])
    quiet = env < thr

    # interior runs only — never touch file edges
    runs = []
    i = 1
    while i < n - 1:
        if quiet[i]:
            j = i
            while j < n - 1 and quiet[j]:
                j += 1
            if (j - i) * 10.0 >= max_ms:
                runs.append((i * h, min(j * h, len(x))))
            i = j
        else:
            i += 1
    if not runs:
        return x

    keep_n = int(tgt_ms * fs / 1000.0)
    fade_n = int(max(1, fade_ms * fs / 1000.0))

    def shorten(a, b):
        """Shorten gap [a, b): keep the first keep_n samples, then blend
        into the resumed audio.  Returns (shortened_audio, samples_consumed)
        so the caller can resume exactly where the crossfade ends."""
        keep = min(keep_n, max(fade_n * 2, (b - a) - fade_n))
        head = x[a:a + keep]
        m = min(fade_n, len(head), max(0, len(x) - b))
        if m > 1:
            w = np.linspace(0.0, 1.0, m)
            joint = head[-m:] * (1.0 - w) + x[b:b + m] * w
            return np.concatenate([head[:-m], joint]), m
        return head, 0

    parts = [x[:runs[0][0]]]
    for k, (a, b) in enumerate(runs):
        short, consumed = shorten(a, b)
        parts.append(short)
        nxt = runs[k + 1][0] if k + 1 < len(runs) else len(x)
        parts.append(x[b + consumed:nxt])
    y = np.concatenate([p for p in parts if len(p)])
    return y.astype(x.dtype) if len(y) else x


def _emphasis_env(x, fs, word_times, emph_words, boost_db=2.5, ramp_factor=0.35):
    if not emph_words or not word_times:
        return x
    want = list(emph_words)
    ranges = []
    wi = 0
    norm = lambda s: re.sub(r"[^\w'’]", "", s.lower())
    for (t0, dur, txt) in word_times:
        if wi < len(want) and norm(txt) == want[wi]:
            ranges.append((t0, t0 + dur))
            wi += 1
    if not ranges:
        return x
    env = np.ones(len(x))
    g = 10 ** (boost_db / 20.0)
    for (t0, t1) in ranges:
        a, b = max(0, int(t0 * fs)), min(len(x), int(t1 * fs))
        if b <= a + 4:
            continue
        ramp = max(1, int((b - a) * ramp_factor))
        shape = np.ones(b - a)
        shape[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, ramp))
        shape[-ramp:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1, 0, ramp))
        env[a:b] = np.maximum(env[a:b], 1.0 + (g - 1.0) * shape)
    return x * env


def _breath(fs, rng, ref_rms, emotion, config: Optional[Dict] = None, exhale=False, size=1.0):
    cfg = config or {}
    dur = rng.uniform(cfg.get("duration_min", 0.15), cfg.get("duration_max", 0.26)) * max(0.7, min(size, 1.6))
    n = int(fs * dur)
    if n < 64:
        return None
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(5) / 5.0, mode="same")
    noise = np.convolve(noise, np.ones(17) / 17.0, mode="same")
    t = np.linspace(0.0, 1.0, n)
    if exhale:
        env = np.minimum(t / 0.10, 1.0) * np.exp(-3.4 * t)
        env *= 1.0 + 0.30 * np.sin(2.0 * np.pi * 2.6 * t)
    else:
        env = np.sin(np.pi * t) ** 1.25
    b = noise * env
    rms = np.sqrt((b ** 2).mean()) + 1e-12
    base = cfg.get("level_normal", 0.14) if emotion not in ("whisper", "tired", "gentle") else cfg.get("level_emotional", 0.18)
    lvl = ref_rms * (base * (0.6 if exhale else 1.0))
    b *= lvl / rms
    nf = min(len(b), max(1, int(fs * 0.006)))
    b[:nf] *= np.linspace(0, 1, nf)
    b[-nf:] *= np.linspace(1, 0, nf)
    return b.astype(np.float64)


def _word_life(x, fs, word_times, rng, db=1.7):
    if not word_times:
        return x
    env = np.ones(len(x))
    for (t0, dur, txt) in word_times:
        g = 10.0 ** (rng.uniform(-db, db) / 20.0)
        a, b = max(0, int(t0 * fs)), min(len(x), int((t0 + dur) * fs))
        if b <= a + 4:
            continue
        ramp = max(1, int((b - a) * 0.30))
        shape = np.ones(b - a)
        shape[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, ramp))
        shape[-ramp:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1, 0, ramp))
        env[a:b] *= 1.0 + (g - 1.0) * shape
    return x * env


def _phrase_swell(x, fs, rng, db=2.6):
    if len(x) < fs // 2:
        return x
    m = max(8, int(len(x) / fs * 20))
    w = rng.standard_normal(m)
    w = np.convolve(w, np.ones(5) / 5.0, mode="same")
    w = w / (np.std(w) + 1e-12)
    env = np.interp(np.linspace(0, m - 1, len(x)), np.arange(m), w)
    return x * (10.0 ** ((db * env) / 20.0))


def _mouth_click(fs: int, rng, duration_ms: float = 6.0) -> np.ndarray:
    """
    Unit-RMS micro transient simulating a natural lip parting / mouth click
    at the start of a phrase.  Config: 'mouth_click' in voice_config.yaml.
    """
    n = max(8, int(fs * np.clip(duration_ms, 2.0, 20.0) / 1000.0))
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(3) / 3.0, mode="same")   # softens harshness
    t = np.linspace(0.0, 1.0, n)
    env = np.exp(-7.0 * t) * (1.0 - np.exp(-40.0 * t))          # fast crack, fast decay
    c = noise * env
    rms = float(np.sqrt(np.mean(c ** 2)) + 1e-12)
    return (c / rms).astype(np.float64)


def _keyword_emphasis(x: np.ndarray, fs: int, words, rng,
                      key_words, cfg: Optional[Dict] = None,
                      low_words=None) -> np.ndarray:
    """
    Gentle emphasis on content words — the audible result of higher
    subglottic pressure on important words — and a matching softening on
    hedging words.  Words already emphasised via **markers** are skipped to
    avoid double-lifting.  Config: 'keyword_emphasis'.
    """
    cfg = cfg or {}
    if not words or (not key_words and not low_words):
        return x
    boost = float(cfg.get("boost_db", 2.0))
    soften_db = float(cfg.get("soften_db", -1.2))
    if boost <= 0.0 and soften_db >= 0.0:
        return x

    def apply(target, db):
        keys = {str(k).lower() for k in target}
        env = np.ones(len(x))
        hit = False
        for (t0, dur, txt) in words:
            if str(txt).lower().strip(".,!?;:") not in keys:
                continue
            a, b = max(0, int(t0 * fs)), min(len(x), int((t0 + dur) * fs))
            if b <= a + 4:
                continue
            g = 10.0 ** (db * rng.uniform(0.6, 1.0) / 20.0)
            ramp = max(1, int((b - a) * 0.35))
            shape = np.ones(b - a)
            shape[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, ramp))
            shape[-ramp:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1, 0, ramp))
            env[a:b] *= 1.0 + (g - 1.0) * shape
            hit = True
        return x * env if hit else x

    if key_words:
        x = apply(key_words, boost)
    if low_words and soften_db < 0.0:
        x = apply(low_words, soften_db)
    return x


def _emphasis_tilt(x: np.ndarray, fs: int, word_times, emph_words,
                   cfg: Optional[Dict] = None) -> np.ndarray:
    """
    Emphasis brightness (Phase C): a gentle high-shelf lift applied only on
    emphasised words, so an accent is heard as brighter *and* louder rather
    than just a level bump.  The shelf is computed over the whole take with
    the phase-preserving biquad EQ, then blended in under a smooth per-word
    envelope — no per-sample filtering, no seams.  Config: 'emphasis.brightness'.
    """
    cfg = cfg or {}
    if not cfg.get("enabled", False) or not emph_words or not word_times:
        return x
    tilt_db = float(cfg.get("tilt_db", 0.0))
    if abs(tilt_db) < 1e-3:
        return x
    hz = float(cfg.get("hz", 3800.0))
    q = float(cfg.get("q", 0.9))
    ramp_factor = float(cfg.get("ramp_factor", 0.35))

    def norm(s):
        return re.sub(r"[^\w'’]", "", s.lower())

    want = [norm(w) for w in emph_words]
    ranges = []
    wi = 0
    for (t0, dur, txt) in word_times:
        if wi < len(want) and norm(txt) == want[wi]:
            ranges.append((t0, t0 + dur))
            wi += 1
    if not ranges:
        return x

    bright = _eq(x.astype(np.float64), fs, [("pk", hz, tilt_db, q)])
    env = np.zeros(len(x))
    for (t0, t1) in ranges:
        a, b = max(0, int(t0 * fs)), min(len(x), int(t1 * fs))
        if b <= a + 4:
            continue
        ramp = max(1, int((b - a) * ramp_factor))
        shape = np.ones(b - a)
        shape[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, ramp))
        shape[-ramp:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1, 0, ramp))
        env[a:b] = np.maximum(env[a:b], shape)
    if not env.any():
        return x
    return (x + env * (bright - x)).astype(x.dtype)


def _de_accent(x: np.ndarray, fs: int, word_times, rng, given_words,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """
    Given-information de-accenting (Phase B): words already introduced earlier
    in the script ('given' information) are delivered with slightly less
    prominence than genuinely new information.  A gentle, ramped level cut on
    those words only.  Emphasis always wins: callers exclude emphasised /
    content-boosted words from `given_words`.
    Config: 'de_accent'.
    """
    cfg = cfg or {}
    if not cfg.get("enabled", False) or not given_words or not word_times:
        return x
    soften_db = float(cfg.get("given_soften_db", -1.5))
    if soften_db >= 0.0:
        return x
    keys = {str(w).lower() for w in given_words}
    env = np.ones(len(x))
    hit = False
    for (t0, dur, txt) in word_times:
        if str(txt).lower().strip(".,!?;:") not in keys:
            continue
        a, b = max(0, int(t0 * fs)), min(len(x), int((t0 + dur) * fs))
        if b <= a + 4:
            continue
        g = 10.0 ** (soften_db * rng.uniform(0.6, 1.0) / 20.0)
        ramp = max(1, int((b - a) * 0.35))
        shape = np.ones(b - a)
        shape[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, ramp))
        shape[-ramp:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1, 0, ramp))
        env[a:b] *= 1.0 + (g - 1.0) * shape
        hit = True
    return x * env if hit else x


def _breath_before_keyword(x: np.ndarray, fs: int, words, rng,
                           ref_rms: float, emotion,
                           breath_cfg: Dict,
                           key_words) -> np.ndarray:
    """
    Insert a soft indrawn breath just before the first emphasised word in a
    phrase.  Returns the extended signal (word times are handled by the
    caller's ordering).
    """
    if not key_words or not words:
        return x
    keys = {str(k).lower() for k in key_words}
    target = None
    for (t0, dur, txt) in words:
        if str(txt).lower().strip(".,!?;:") in keys and t0 > 0.05:
            target = float(t0)
            break
    if target is None:
        return x
    br = _breath(fs, rng, ref_rms, emotion, config=breath_cfg, size=0.55)
    if br is None:
        return x
    pos = int(target * fs)
    pos = max(0, min(len(x) - 1, pos))
    return np.concatenate([x[:pos], br.astype(x.dtype), x[pos:]])


def _ola_expand(seg: np.ndarray, fs: int, factor: float) -> np.ndarray:
    """Pitch-preserving time expansion via overlap-add frame repetition."""
    win = max(64, int(fs * 0.025))
    if len(seg) < win * 2 or factor <= 1.001:
        return seg
    hop_in  = max(1, win // 2)
    hop_out = max(1, int(round(hop_in * factor)))
    win_fn  = np.hanning(win)
    out_len = int(len(seg) * factor) + win
    out     = np.zeros(out_len)
    norm    = np.zeros(out_len)
    pos = i = 0
    while pos < len(seg):
        fl = min(win, len(seg) - pos)
        o  = i * hop_out
        if o + fl > out_len:
            break
        out[o:o + fl]  += seg[pos:pos + fl] * win_fn[:fl]
        norm[o:o + fl] += win_fn[:fl]
        pos += hop_in
        i   += 1
    nz = norm > 1e-9
    out[nz] /= norm[nz]
    return out[:int(len(seg) * factor)]


def _elongate_words(x: np.ndarray, fs: int, words, rng,
                    cfg: Optional[Dict] = None):
    """
    Selectively stretch individual words (vowel elongation while thinking).
    Pitch-preserving OLA expansion; word timestamps are kept in sync.
    Config: 'word_elongation' in voice_config.yaml.
    """
    cfg = cfg or {}
    if not words or len(x) < fs // 4:
        return x, words
    prob      = float(cfg.get("probability", 0.15))
    f_lo      = float(cfg.get("min_factor", 1.12))
    f_hi      = float(cfg.get("max_factor", 1.30))
    min_chars = int(cfg.get("min_chars", 3))

    new_words = list(words)
    for wi in range(len(new_words)):
        t0, dur, txt = new_words[wi]
        if dur < 0.06 or len(str(txt).strip()) < min_chars:
            continue
        if rng.random() >= prob:
            continue
        factor = float(rng.uniform(f_lo, f_hi))
        a = max(0, int(t0 * fs))
        b = min(len(x), int((t0 + dur) * fs))
        if b - a < int(fs * 0.05):
            continue
        stretched = _ola_expand(x[a:b].astype(np.float64), fs, factor)
        # Short edge fades keep word boundaries from clicking (clarity)
        edge = min(int(fs * 0.008), max(1, len(stretched) // 4))
        stretched[:edge]  *= np.linspace(0.0, 1.0, edge)
        stretched[-edge:] *= np.linspace(1.0, 0.0, edge)
        shift = (len(stretched) - (b - a)) / fs
        x = np.concatenate([x[:a], stretched, x[b:]]).astype(x.dtype)
        new_words[wi] = (t0, len(stretched) / fs, txt)
        for wj in range(wi + 1, len(new_words)):
            wt, wd, wtx = new_words[wj]
            new_words[wj] = (wt + shift, wd, wtx)
    return x, new_words


# --------------------------------------------------------------------------- #
# Dynamic text analysis — fully config-driven, zero hardcoded word lists       #
# --------------------------------------------------------------------------- #

def _analyze_text_dynamics(text: str, config: Optional[Dict] = None) -> Dict:
    """
    Analyse text to compute dynamic prosody adjustments.
    All thresholds and word-lists live in voice_config.yaml under
    'dynamic_prosody'. Nothing is hardcoded here.
    """
    cfg = (config or {}).get("dynamic_prosody", {})
    if not cfg.get("enabled", True):
        return {}

    text_lower = text.lower()
    words = re.findall(r"[\w'\u0900-\u097F]+", text_lower)  # supports Hindi too
    if not words:
        return {}

    iw       = cfg.get("intensity_words", {})
    high_set = set(iw.get("high",   []))
    med_set  = set(iw.get("medium", []))
    low_set  = set(iw.get("low",    []))

    intensity = 0.0
    key_words = []
    low_words = []
    for w in words:
        if   w in high_set:
            intensity += 1.0
            key_words.append(w)
        elif w in med_set:
            intensity += 0.5
            key_words.append(w)
        elif w in low_set:
            intensity -= 0.3
            low_words.append(w)

    # Normalise to [-1, 1] proportional to sentence length
    intensity = float(np.clip(intensity / max(len(words), 1) * 6.0, -1.0, 1.0))

    # Sentence-length → rate nudge (short = snappier, long = more deliberate)
    n_chars     = len(text)
    short_thr   = cfg.get("short_sentence_chars", 30)
    long_thr    = cfg.get("long_sentence_chars",  80)
    rate_delta  = cfg.get("length_rate_delta", 2)
    if   n_chars < short_thr: length_rate =  rate_delta
    elif n_chars > long_thr:  length_rate = -rate_delta
    else:                     length_rate =  0.0

    sp = cfg.get("intensity_pitch_scale", 2.0)
    sr = cfg.get("intensity_rate_scale",  1.5)
    # Vocal effort (subglottic pressure): strong content projects more energy,
    # hedging / casual wording softens.
    sv = cfg.get("intensity_volume_scale", 4.0)

    # ---- Emotional valence (smile / warmth / softening) -----------------
    vw = cfg.get("valence_words", {}) or {}
    pos_set    = set(vw.get("positive", []))
    tender_set = set(vw.get("tender",   []))
    neg_set    = set(vw.get("negative", []))
    pos = tend = neg = 0
    for w in words:
        if   w in pos_set:    pos += 1.0
        elif w in tender_set: tend += 1.0
        elif w in neg_set:    neg += 1.0
    denom = max(len(words), 1)
    smile   = float(np.clip(pos / denom * 8.0, 0.0, 1.0))    # 0..1
    soften  = float(np.clip(tend / denom * 8.0, 0.0, 1.0))   # 0..1
    valence = smile - 0.8 * float(np.clip(neg / denom * 8.0, 0.0, 1.0))

    s_pitch = cfg.get("smile_pitch_st", 1.4)      # semitones of smile lift
    s_vol   = cfg.get("smile_volume_db", -1.2)   # softening on warm phrases
    s_rate  = cfg.get("smile_rate_pct", -1.5)    # warm phrases ease slightly
    s_pause = cfg.get("valence_pause_scale", 0.14)
    # Sweetness / warmth scale with the mood of THIS sentence rather than
    # being locked to the emotion preset.
    s_bright = cfg.get("smile_bright_db", 1.2)
    s_warm   = cfg.get("soften_warm_db", 0.5)
    neg = float(np.clip(neg / denom * 8.0, 0.0, 1.0))
    bright_delta = smile * s_bright + max(intensity, 0.0) * 0.4
    warm_delta   = soften * s_warm - neg * 0.4

    # Breathing interval adapts: more emphatic, warm or substantial passages
    # breathe more often and more deeply.
    length_norm = float(np.clip(len(text) / 160.0, 0.0, 1.0))
    breath_scale = float(np.clip(
        0.75 + 0.45 * max(intensity, 0.0) + 0.40 * soften + 0.30 * length_norm,
        0.4, 1.8))

    return {
        "intensity":     intensity,
        "pitch_delta":   intensity * sp + smile * s_pitch,
        "rate_delta":    intensity * sr + length_rate + (soften + smile) * s_rate,
        "volume_delta":  intensity * sv + (soften * 0.6 + smile) * s_vol,
        "key_words":     key_words,
        "low_words":     low_words,
        "effort":        float(np.clip(intensity, -1.0, 1.0)),
        "valence":       smile - 0.8 * neg,
        "smile":         smile,
        "soften":        soften,
        "pause_scale":   1.0 + (smile - 0.8 * neg) * s_pause,
        "bright_delta":  bright_delta,
        "warm_delta":    warm_delta,
        "breath_scale":  breath_scale,
    }


# --------------------------------------------------------------------------- #
# Natural intonation arc (pitch contour within each phrase)                     #
# --------------------------------------------------------------------------- #

def _build_emotion_contour(n_frames: int, emotion_key: str,
                           kind: Optional[str],
                           config: Optional[Dict] = None) -> np.ndarray:
    """
    Return a semitone contour of length n_frames for a natural rise-fall arc.
    Shape and magnitude are fully parameterised per emotion in voice_config.yaml
    under 'pitch_contour.emotion_modulation'. Questions get a terminal upswing.
    """
    cfg = (config or {}).get("pitch_contour", {})
    if not cfg.get("enabled", False) or n_frames < 4:
        return np.zeros(n_frames)

    mods = cfg.get("emotion_modulation", {})
    mod  = mods.get(emotion_key, mods.get("neutral", {}))

    rise     = float(mod.get("rise", 0.7))
    fall     = float(mod.get("fall", 1.0))
    peak_pos = float(np.clip(mod.get("peak", 0.38), 0.05, 0.95))

    t       = np.linspace(0.0, 1.0, n_frames)
    contour = np.zeros(n_frames)

    # Rising phase: 0 → peak_pos
    m_rise           = t <= peak_pos
    t_rise           = t[m_rise] / peak_pos
    contour[m_rise]  = rise * np.sin(0.5 * np.pi * t_rise)

    # Falling phase: peak_pos → 1
    m_fall  = ~m_rise
    t_fall  = (t[m_fall] - peak_pos) / max(1.0 - peak_pos, 1e-6)
    contour[m_fall] = (rise * np.cos(0.5 * np.pi * t_fall)
                       - fall * np.sin(0.5 * np.pi * t_fall))

    # Questions: add a terminal rise in the last 15% of the phrase
    if kind == "question":
        q_rise = cfg.get("question_terminal_rise_st", 1.2)
        last   = max(1, int(n_frames * 0.15))
        contour[-last:] += q_rise * np.linspace(0.0, 1.0, last)

    return contour


def _apply_pitch_arc(x: np.ndarray, fs: int,
                     contour_st: np.ndarray,
                     hop_ms: float = 20.0) -> np.ndarray:
    """
    Apply a time-varying pitch contour (semitones) via overlap-add.

    Each analysis window is resampled to the required length and the CENTRE
    of the resampled frame is windowed back to its original size.  The older
    implementation zero-padded short frames, which punched amplitude holes
    into the signal and was heard as a 'shiver' between phrases.
    Duration is preserved; only pitch moves.
    """
    if np.max(np.abs(contour_st)) < 0.05:
        return x
    try:
        from scipy.signal import resample
    except ImportError:
        return x

    hop   = max(1, int(fs * hop_ms / 1000.0))
    win_n = hop * 4
    if len(x) < win_n * 2:
        return x

    n_frames = max(1, (len(x) - win_n) // hop)
    ci = np.interp(np.linspace(0, 1, n_frames),
                   np.linspace(0, 1, len(contour_st)),
                   contour_st)

    win_fn = np.hanning(win_n)
    out_len = len(x) + win_n
    out  = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)

    xf = x.astype(np.float64)
    for i in range(n_frames):
        s = i * hop
        frame = xf[s:s + win_n]
        if len(frame) < win_n:
            break
        r = 2.0 ** (ci[i] / 12.0)          # >1 raises pitch
        if abs(r - 1.0) < 1e-4:
            out[s:s + win_n]  += frame * win_fn
            norm[s:s + win_n] += win_fn
            continue
        m = max(16, int(round(win_n * r)))
        fr = resample(frame, m)
        off = (m - win_n) // 2
        if off < 0:
            fr = np.concatenate([fr, np.zeros(-off)])
            off = 0
        fr = fr[off:off + win_n]
        if len(fr) < win_n:
            fr = np.concatenate([fr, np.zeros(win_n - len(fr))])
        out[s:s + win_n]  += fr[:win_n] * win_fn
        norm[s:s + win_n] += win_fn

    nz = norm > 1e-9
    out[nz] /= norm[nz]
    result = out[:len(x)]

    rms_in  = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(result ** 2)) + 1e-12)
    if rms_out > 1e-9:
        result *= rms_in / rms_out
    return result.astype(x.dtype)


# --------------------------------------------------------------------------- #
# Micro-prosody — natural shimmer + jitter (config-driven)                      #
# --------------------------------------------------------------------------- #

def _apply_micro_prosody(x: np.ndarray, fs: int, rng,
                         config: Optional[Dict] = None) -> np.ndarray:
    """
    Natural vocal amplitude shimmer (slow amplitude modulation at voicing rate).
    Parameters are fully config-driven from 'micro_prosody' in voice_config.yaml.
    """
    cfg = config or {}
    if not cfg.get("enabled", False):
        return x

    shimmer_db = cfg.get("amplitude_shimmer_db", 0.8)
    rate_hz    = cfg.get("rate_hz", 6.0)
    n          = len(x)
    t          = np.arange(n) / fs
    depth      = (shimmer_db / 20.0) * np.log(10.0)

    # Primary shimmer (vocal fold amplitude variation ~4-8 Hz)
    ph1     = rng.uniform(0, 2 * np.pi)
    shimmer = 1.0 + depth * np.sin(2 * np.pi * rate_hz * t + ph1)

    # Secondary slower modulation for added realism
    ph2      = rng.uniform(0, 2 * np.pi)
    shimmer += 0.4 * depth * np.sin(2 * np.pi * (rate_hz * 0.38) * t + ph2)

    return (x * shimmer).astype(x.dtype)


def _apply_vocal_fry(x: np.ndarray, fs: int, rng,
                     config: Optional[Dict] = None,
                     strength: float = 1.0) -> np.ndarray:
    """
    Subtle creaky voice (vocal fry) on the tail of a phrase: an irregular
    low-rate amplitude pulse train that eases in over the last portion of
    the segment.  Intensity, tail length and pulse rate come from
    'vocal_fry' in voice_config.yaml — zero hardcoded values.
    """
    cfg = config or {}
    if not cfg.get("enabled", False) or len(x) < int(fs * 0.4):
        return x

    frac    = float(np.clip(cfg.get("duration_frac", 0.16), 0.05, 0.5))
    max_s   = float(cfg.get("max_seconds", 0.40))
    depth   = float(np.clip(cfg.get("intensity", 0.18) * max(0.0, strength), 0.0, 0.6))
    pulse_hz = float(cfg.get("pulse_hz", 34.0))

    tail_n = int(min(len(x) * frac, fs * max_s))
    if tail_n < int(fs * 0.06) or depth <= 0.0:
        return x

    start = len(x) - tail_n
    t_end = tail_n / fs

    # Irregular jittered pulse train — organic creak, not a steady buzzer
    pulses = np.zeros(tail_n)
    pos = rng.uniform(0.0, 1.0 / max(pulse_hz, 8.0))
    while pos < t_end:
        period = (1.0 / max(pulse_hz, 8.0)) * rng.uniform(0.7, 1.45)
        p_len = max(2, int(period * fs * 0.42))
        i0 = int(pos * fs)
        if i0 >= tail_n:
            break
        n = min(p_len, tail_n - i0)
        shape = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.7
        pulses[i0:i0 + n] = np.maximum(pulses[i0:i0 + n], shape)
        pos += period

    # Effect eases in over the tail, strongest at the very end
    ramp = np.linspace(0.0, 1.0, tail_n) ** 1.5
    env = 1.0 - depth * ramp * (1.0 - pulses)

    y = x.astype(np.float64).copy()
    y[start:] *= env
    return y.astype(x.dtype)


def _estimate_f0(x, fs, sample_seconds=6.0, fmin=70.0, fmax=350.0):
    """
    Cheap median-F0 estimate (Hz) over a spread of the signal.

    Used by the pitch guard to verify the rendered voice actually sounds like
    the intended speaker, and to correct it if a phrase came out too low/high.
    Returns 0.0 when nothing is confidently voiced.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < fs // 4:
        return 0.0
    hop = int(fs * 0.01)
    win = int(fs * 0.04)
    # sample at most `sample_seconds` spread across the whole utterance
    max_frames = int(sample_seconds / 0.01)
    total = max(1, (len(x) - win) // hop)
    step = max(1, total // max_frames)
    lags = np.arange(int(fs / fmax), int(fs / fmin))
    vals = []
    for i in range(0, total, step):
        fr = x[i * hop:i * hop + win]
        if len(fr) < win:
            break
        if np.sqrt(np.mean(fr ** 2)) < 0.03:
            continue
        fr = fr * np.hanning(win)
        ac = np.correlate(fr, fr, "full")[win - 1:]
        ac = ac / (ac[0] + 1e-12)
        L = lags[int(np.argmax(ac[lags]))]
        if ac[L] > 0.4:
            vals.append(fs / L)
    if len(vals) < 5:
        return 0.0
    v = np.asarray(vals, dtype=np.float64)
    med = float(np.median(v))
    if med <= 0:
        return 0.0
    # Fold octave errors against the median (vectorised — a scalar loop here
    # silently fails to modify the array and hides genuinely low pitch).
    for _ in range(4):
        v = np.where(v > med * 1.45, v * 0.5, v)
        v = np.where(v < med * 0.68, v * 2.0, v)
    return float(np.median(v))


def _follow_compress(x, fs, thr_db=-21.0, ratio=2.0, atk=0.008, rel=0.16):
    if ratio <= 1.001:          # zero compression — bypass entirely
        return x
    ax = np.abs(x)
    aA = np.exp(-1.0 / (atk * fs))
    aR = np.exp(-1.0 / (rel * fs))
    e = 1e-9
    gains = np.empty(len(x))
    g_cur = 1.0
    gA = np.exp(-1.0 / (0.004 * fs))
    gR = np.exp(-1.0 / (0.060 * fs))
    for i in range(len(x)):
        v = ax[i]
        if v > e:
            e = aA * e + (1.0 - aA) * v
        else:
            e = aR * e + (1.0 - aR) * v
        edb = 20.0 * np.log10(e + 1e-9)
        over = edb - thr_db
        g = 10.0 ** (-over * (1.0 - 1.0 / ratio) / 20.0) if over > 0 else 1.0
        if g < g_cur:
            g_cur = gA * g_cur + (1.0 - gA) * g
        else:
            g_cur = gR * g_cur + (1.0 - gR) * g
        gains[i] = g_cur
    return x * gains


def _smooth_env(x, fs, win_s=0.020):
    w = max(1, int(fs * win_s))
    k = np.ones(w) / w
    return np.sqrt(np.convolve(x ** 2, k, mode="same"))


# --------------------------------------------------------------------------- #
# Mastering                                                                    #
# --------------------------------------------------------------------------- #

def _up2x(x, fs, target=SR_OUT):
    x = np.asarray(x, dtype=np.float32)
    if fs == target:
        return x, fs
    if fs == 24000 and target == 48000:
        n = len(x)
        X = np.fft.rfft(x)
        weights = np.ones(len(X))
        rolloff = max(1, int(len(X) * 0.08))
        weights[-rolloff:] = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, rolloff)))
        Y = np.zeros(n + 1, dtype=complex)
        Y[:len(X)] = X * weights
        y = np.fft.irfft(Y, 2 * n) * 2.0
        return y.astype(np.float32), target
    n_out = max(2, int(round(len(x) * float(target) / float(fs))))
    y = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    return y.astype(np.float32), target


def _pink_noise(n, rng):
    """Generate pink noise using Voss-McCartney algorithm."""
    n_rows = 16
    cols = n // n_rows
    array = np.empty((n_rows, cols))
    array.fill(np.nan)
    array[0, :] = rng.standard_normal(cols)
    array[-1, :] = rng.standard_normal(cols)
    s = np.arange(1, n_rows - 1)
    array[s, :] = rng.standard_normal((len(s), cols))
    pink = np.zeros(n_rows * cols)
    for i in range(n_rows):
        col = array[i, :]
        col = np.nan_to_num(col, nan=0.0)
        pink[i::n_rows] = col[:len(pink[i::n_rows])]
    pink = np.convolve(pink, np.ones(8) / 8.0, mode="same")
    if len(pink) < n:
        pink = np.pad(pink, (0, n - len(pink)), mode="edge")
    else:
        pink = pink[:n]
    return pink


def _room_tone(fs, duration, rng, intensity_db=-48, config: Optional[Dict] = None):
    """Generate realistic room tone with pink noise and subtle modulation."""
    cfg = config or {}
    n = int(fs * duration)
    pink = _pink_noise(n, rng)

    lp_hz = cfg.get("lowpass_hz", 800.0)
    lp_q = cfg.get("lowpass_q", 0.5)
    b, a = _lp(fs, lp_hz, q=lp_q)
    pink_filt = _bj_mag(b, a, len(pink), fs)
    pink_fft = np.fft.rfft(pink) * pink_filt
    pink = np.fft.irfft(pink_fft, n)

    t = np.arange(n) / fs
    m1_hz = cfg.get("modulation_1_hz", 0.3)
    m1_amp = cfg.get("modulation_1_amp", 0.15)
    m2_hz = cfg.get("modulation_2_hz", 0.7)
    m2_amp = cfg.get("modulation_2_amp", 0.08)
    m3_hz = cfg.get("modulation_3_hz", 1.3)
    m3_amp = cfg.get("modulation_3_amp", 0.05)
    mod1 = m1_amp * np.sin(2 * np.pi * m1_hz * t + rng.uniform(0, np.pi))
    mod2 = m2_amp * np.sin(2 * np.pi * m2_hz * t + rng.uniform(0, np.pi))
    mod3 = m3_amp * np.sin(2 * np.pi * m3_hz * t + rng.uniform(0, np.pi))
    envelope = 1.0 + mod1 + mod2 + mod3

    pink *= envelope

    hum50 = cfg.get("hum_50hz", 0.02)
    hum100 = cfg.get("hum_100hz", 0.01)
    hum = hum50 * np.sin(2 * np.pi * 50.0 * t) + hum100 * np.sin(2 * np.pi * 100.0 * t)
    pink += hum

    rms = np.sqrt(np.mean(pink ** 2)) + 1e-12
    pink *= (10 ** (intensity_db / 20.0)) / rms

    return pink


def _air_breath(fs, duration, rng, intensity_db=-52, config: Optional[Dict] = None):
    """Subtle air movement sound."""
    cfg = config or {}
    n = int(fs * duration)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(12) / 12.0, mode="same")

    bp_hz = cfg.get("lowpass_hz", 400.0)
    bp_bw = cfg.get("bandwidth", 200.0)
    b, a = _bp(fs, bp_hz, bp_bw)
    noise_filt = _bj_mag(b, a, len(noise), fs)
    noise_fft = np.fft.rfft(noise) * noise_filt
    noise = np.fft.irfft(noise_fft, n)

    t = np.arange(n) / fs
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.7
    noise *= env

    rms = np.sqrt(np.mean(noise ** 2)) + 1e-12
    noise *= (10 ** (intensity_db / 20.0)) / rms
    return noise


def _bp(fs, fc, bw, q=None):
    if q is None:
        q = fc / bw
    w0 = 2.0 * np.pi * fc / fs
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b0 = al
    b = [b0, 0.0, -b0]
    a0 = 1.0 + al
    a = [1.0, (-2.0 * c) / a0, (1.0 - al) / a0]
    b = [x / a0 for x in b]
    return b, a


def master(track, fs, em, persona: Persona, rng, config: Optional[Dict] = None,
           emotion_key: Optional[str] = None):
    cfg = config or {}
    room_cfg = cfg.get("room_tone", {})
    air_cfg = cfg.get("air_breath", {})

    x = track.astype(np.float64)
    env = _smooth_env(x, fs)
    thr = np.percentile(env, 55)

    # Phase F — room-tone whisper.  Room tone is otherwise off; for a whisper
    # delivery a low bed of room tone / air is exactly what sells the
    # "close to the mic in a quiet room" character, so it can be enabled for
    # whisper alone via 'whisper_room_tone'.
    wrt_cfg = cfg.get("whisper_room_tone", {}) or {}
    is_whisper = emotion_key == "whisper"
    room_on = room_cfg.get("enabled", False)
    whisper_room_on = is_whisper and wrt_cfg.get("enabled", False)

    if room_on or whisper_room_on:
        if thr < 1e-4:
            # No measurable noise floor (e.g. a synthesised track with hard
            # silence): nothing to gate against, so let the bed sit under the
            # whole take rather than collapsing to zero.
            gate = np.ones_like(env)
        else:
            gate = np.clip((thr * 1.1 - env) / thr, 0.0, 1.0)
        gw = np.hanning(max(3, int(fs * 0.05)))
        gw /= gw.sum()
        gate = np.convolve(gate, gw, mode="same")
        duration = len(x) / fs
        if whisper_room_on and not room_on:
            rc = {**room_cfg, **{k: v for k, v in wrt_cfg.items() if k != "enabled"}}
            level = float(wrt_cfg.get("level_db", em["air"]))
        else:
            rc = room_cfg
            level = em["air"]
        room = _room_tone(fs, duration, rng, intensity_db=level, config=rc)
        air_offset = air_cfg.get("intensity_offset", -4)
        air_breath = _air_breath(fs, duration, rng, intensity_db=level + air_offset, config=air_cfg)
        room_ratio = rc.get("room_ratio", 0.7)
        breath_ratio = rc.get("breath_ratio", 0.3)
        min_len = min(len(room), len(air_breath), len(x))
        combined_air = room[:min_len] * room_ratio + air_breath[:min_len] * breath_ratio
        x[:min_len] += combined_air * gate[:min_len]

    x = _follow_compress(x, fs, thr_db=persona.target_rms_db - 1.0, ratio=em["comp"])

    env = _smooth_env(x, fs)
    active = env > max(thr, 1e-6) * 0.5
    sp_rms = np.sqrt(np.mean(x[active] ** 2)) if active.any() else np.sqrt(np.mean(x ** 2))
    x *= (10 ** (persona.target_rms_db / 20.0)) / max(sp_rms, 1e-9)

    ceil = 10 ** (persona.peak_ceiling_db / 20.0)
    if np.max(np.abs(x)) > ceil:
        # Soft-knee limiter: bit-exact below the knee, asymptotic to the
        # ceiling above it.  (The previous tanh normaliser overshot the
        # ceiling by ~3 dB and coloured the whole signal.)
        knee = ceil * float(cfg.get("persona", {}).get("limiter_knee", 0.7))
        knee = min(max(knee, 0.1), 0.95) * ceil
        span = max(ceil - knee, 1e-9)
        over = np.abs(x) > knee
        x = x.copy()
        x[over] = np.sign(x[over]) * (knee + span * np.tanh((np.abs(x[over]) - knee) / span))
        np.clip(x, -ceil, ceil, out=x)

    x = _fade(x, fs, 8.0)
    y, fs2 = _up2x(x.astype(np.float32), fs, target=int(cfg.get("output", {}).get("sample_rate", SR_OUT)))
    # Resampling can overshoot the ceiling, so the peak limit is enforced
    # again on the final sample rate.
    ceil2 = 10 ** (persona.peak_ceiling_db / 20.0)
    peak2 = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak2 > ceil2:
        y = (y / peak2) * ceil2
    return y, fs2


# --------------------------------------------------------------------------- #
# Render                                                                       #
# --------------------------------------------------------------------------- #

def _apply_terminal(segments, cfg):
    """Shape the utterance-final segment: gentle fall on declarative endings,
    slight lift on question endings.  Deltas come from
    `sentence_type_rules.<kind>.terminal_pitch` — nothing hardcoded."""
    if not segments:
        return
    last = segments[-1]
    if last.get("is_filler", False):
        return
    t_rule = (cfg or {}).get("sentence_type_rules", {}).get(
        last.get("kind") or "sentence", {})
    t_delta = int(t_rule.get("terminal_pitch", 0))
    if t_delta:
        v = VOICES[last["vkey"]]
        last["pitch"] = int(np.clip(last["pitch"] + t_delta, v["lo"], v["hi"]))


def _insert_fillers(segments, filler_cfg, rng):
    """Phase D — place spoken hesitations ("uhm") inside natural pauses.

    Pure (no audio, no network), so placement can be tested deterministically.
    Returns a new segment list with filler segments spliced into the pauses
    that qualify.  Refinements over a naive insertion:
      * 'split_jitter'      keeps a hesitation from always sitting dead-centre;
      * 'min_spacing_segments' avoids two hesitations back-to-back;
      * 'boundary_kinds'    anchors hesitations to real thought boundaries;
      * 'draw_out'          occasionally draws one out ("ummm").
    """
    tiers = []
    base_words = list(filler_cfg.get("words", []) or [])
    if base_words and float(filler_cfg.get("probability", 0.0)) > 0.0:
        tiers.append(dict(
            words=base_words,
            prob=float(filler_cfg.get("probability", 0.0)),
            min_pause=float(filler_cfg.get("min_pause_ms", 175)),
            gain=float(filler_cfg.get("gain_db", -10.0)),
            rate=str(filler_cfg.get("rate", "-6%")),
            volume=str(filler_cfg.get("volume", "-12%")),
            pitch=int(filler_cfg.get("pitch_hz", -3)),
            max_s=float(filler_cfg.get("max_seconds", 0.6)),
            split=0.5))
    micro = filler_cfg.get("micro", {}) or {}
    if list(micro.get("words", []) or []) and float(micro.get("probability", 0.0)) > 0.0:
        tiers.append(dict(
            words=list(micro["words"]),
            prob=float(micro["probability"]),
            min_pause=float(micro.get("min_pause_ms", 90)),
            gain=float(micro.get("gain_db", -13.0)),
            rate=str(micro.get("rate", "+0%")),
            volume=str(micro.get("volume", "-15%")),
            pitch=int(micro.get("pitch_hz", -4)),
            max_s=float(micro.get("max_seconds", 0.45)),
            split=0.4))
    if not tiers:
        return segments

    placed = []
    last_index = len(segments) - 1
    spacing = int(filler_cfg.get("min_spacing_segments", 0) or 0)
    jitter = float(filler_cfg.get("split_jitter", 0.0) or 0.0)
    _bk = filler_cfg.get("boundary_kinds")
    allowed_kinds = {str(k) for k in _bk} if _bk else None
    draw = filler_cfg.get("draw_out", {}) or {}
    since_filler = spacing  # the first eligible pause may fire
    for si, s in enumerate(segments):
        chosen = None
        # Never place a hesitation AFTER the final segment: there is no
        # following phrase, so it just dangles at the end.
        if (si != last_index and since_filler >= spacing
                and (allowed_kinds is None
                     or s.get("kind") in allowed_kinds)):
            for tier in tiers:
                if (s["pause_ms"] >= tier["min_pause"]
                        and rng.random() < tier["prob"]):
                    chosen = tier
                    break
        if chosen is None:
            since_filler += 1
            placed.append(s)
            continue
        # A hesitation does not always sit dead-centre in the pause; a little
        # jitter keeps it from feeling mechanical.
        split = float(chosen["split"])
        if jitter > 0.0:
            split = float(np.clip(split + rng.uniform(-jitter, jitter), 0.2, 0.8))
        half = s["pause_ms"] * split
        s["pause_ms"] = half
        fseg = dict(s)
        fseg["text"] = str(rng.choice(chosen["words"]))
        fseg["kind"] = "filler"
        fseg["is_filler"] = True
        fseg["emph"] = []
        fseg["key_words"] = []
        fseg["effort"] = 0.0
        fseg["valence"] = 0.0
        fseg["breath_scale"] = 1.0
        fseg["pitch"] = s["pitch"] + chosen["pitch"]
        fseg["rate"] = chosen["rate"]
        fseg["volume"] = chosen["volume"]
        fseg["pause_ms"] = half
        fseg["filler_gain_db"] = chosen["gain"]
        fseg["filler_max_s"] = chosen["max_s"]
        # Drawn-out variant ("ummm"): a slower, longer hesitation.
        if (draw.get("enabled", False)
                and rng.random() < float(draw.get("probability", 0.0))):
            fseg["rate"] = str(draw.get("rate", chosen["rate"]))
            fseg["filler_max_s"] = float(draw.get("max_seconds", chosen["max_s"]))
        placed.append(s)
        placed.append(fseg)
        since_filler = 0
    return placed


def render(text, out_wav, emotion=DEFAULT_EMOTION, persona: Persona = None,
           accent=DEFAULT_ACCENT, seed=20261009, breaths=True, config=None, register=DEFAULT_REGISTER,
           verbatim=False):
    return render_arc([(text, emotion)], out_wav, persona=persona,
                      accent=accent, seed=seed, breaths=breaths, config=config, register=register,
                      verbatim=verbatim)


def render_arc(steps, out_wav, persona: Persona = None,
               accent=DEFAULT_ACCENT, seed=20261009, breaths=True,
               config: Optional[Dict] = None, register: str = DEFAULT_REGISTER,
               verbatim: bool = False):
    persona = persona or Persona()
    rng = np.random.default_rng(seed)
    acc = ACCENTS.get(accent, ACCENTS[DEFAULT_ACCENT])
    cfg = config or {}

    breath_cfg = cfg.get("breath", {})
    emphasis_cfg = cfg.get("emphasis", {})
    persona_cfg = cfg.get("persona", {})
    word_life_cfg = cfg.get("word_life", {})
    phrase_swell_cfg = cfg.get("phrase_swell", {})
    voice_settings_cfg = cfg.get("voice_settings", {})
    variation_cfg = cfg.get("variation", {})

    segments = []
    for text, emo in steps:
        emo_key = emo.value if isinstance(emo, Emotion) else emo
        segs = _segment(text, emo_key, persona, acc["rate"], rng,
                        register=register, variation=variation_cfg, config=cfg,
                        verbatim=verbatim)
        for s in segs:
            s["vkey"] = _pick_voice(s["text"], accent)
        if segments and segs:
            reg_pause = REGISTERS.get(register, REGISTERS["normal"])["pause"]
            segments[-1]["pause_ms"] = max(
                segments[-1]["pause_ms"],
                SENTENCE_PAUSE_MS * EMOTIONS.get(emo_key, EMOTIONS["neutral"])["pause"]
                * reg_pause * 0.9 * rng.uniform(0.9, 1.1))
        segments.extend(segs)

    # ---- Cognitive hesitation fillers (uhm / hmm) inside natural pauses ----
    filler_cfg = cfg.get("pause_filler", {})
    if filler_cfg.get("enabled", False):
        segments = _insert_fillers(segments, filler_cfg, rng)


    vm = cfg.get("voice_mode", {})
    single_mode = vm.get("mode") == "single"
    script_rate = vm.get("script_rate", {}) if single_mode else {}
    for s in segments:
        v = VOICES[s["vkey"]]
        s["pitch"] = int(np.clip(s["pitch"] + v.get("pitch", 0), v["lo"], v["hi"]))
        rate_off = int(v.get("rate", 0))
        if single_mode and script_rate:
            # Same voice, per-script pace trim: the one voice can speak both
            # languages but its natural tempo differs by script.  Keeps the
            # identity fixed while both languages land in the target band.
            key = "devanagari" if _is_hindi(s["text"]) else "latin"
            rate_off += int(script_rate.get(key, 0))
        if rate_off:
            # Per-voice (and per-script) rate offset.
            s["rate"] = _shift_pct(s["rate"], rate_off)

    # ---- Adaptive emotional arc ----------------------------------------
    # A script that builds in feeling should sound like it builds: each
    # segment is nudged by how the emotional level has drifted so far,
    # rather than every sentence resetting to the same neutral delivery.
    dp = cfg.get("dynamic_prosody", {})
    arc_gain_p = float(dp.get("arc_pitch_st", 0.9))
    arc_gain_v = float(dp.get("arc_volume_db", 2.0))
    if arc_gain_p or arc_gain_v:
        vals = [float(s.get("valence", 0.0)) for s in segments]
        run = 0.0
        for i, s in enumerate(segments):
            run += vals[i]
            drift = run / max(i + 1, 1)          # 0..1 emotional build
            s["pitch"] = int(np.clip(s["pitch"] + int(round(drift * arc_gain_p)),
                                     VOICES[s["vkey"]]["lo"], VOICES[s["vkey"]]["hi"]))
            s["arc_volume_db"] = drift * arc_gain_v

    # ---- Terminal intonation ---------------------------------------------
    # Gentle fall on declarative endings, slight lift on question endings.
    # Each take already carries its native edge-tts contour; this only shapes
    # the utterance-final segment so multi-take speech still lands naturally.
    # Applied BEFORE smoothing so the final seam still glides (max step).
    _apply_terminal(segments, cfg)

    # ---- Smooth the pitch trajectory across takes ------------------------
    # Consecutive takes can be given quite different pitch targets, which
    # makes the pitch jump at every seam and is heard as shivering.  Limit
    # the step between neighbours so the pitch glides instead.
    max_step = int(cfg.get("humanize", {}).get("max_pitch_step_hz", 3))
    if max_step > 0:
        for i in range(1, len(segments)):
            prev, cur = segments[i - 1]["pitch"], segments[i]["pitch"]
            if abs(cur - prev) > max_step:
                cur = prev + (max_step if cur > prev else -max_step)
            v = VOICES[segments[i]["vkey"]]
            segments[i]["pitch"] = int(np.clip(cur, v["lo"], v["hi"]))

    # ---- Given-information tracking (Phase B) ----------------------------
    # Content words that appeared in an earlier take are 'given'; genuinely
    # new words carry the information.  Emphasis always wins, so words that
    # are explicitly emphasised or content-boosted are never de-accented.
    da_cfg = cfg.get("de_accent", {})
    if da_cfg.get("enabled", False):
        minor = {str(w).lower() for w in (da_cfg.get("minor_words", []) or [])}
        seen = set()
        for s in segments:
            toks = re.findall(r"[\w'\u0900-\u097F]+", s["text"].lower())
            content = [w for w in toks if w not in minor and len(w) > 1]
            skip = {w.lower() for w in (list(s.get("emph", [])) +
                                        list(s.get("key_words", [])))}
            given = [w for w in content if w in seen and w not in skip]
            s["given_words"] = list(dict.fromkeys(given))
            seen.update(content)

    raw = asyncio.run(_edge_all(segments, acc["voice"],
                                cache_cfg=cfg.get("backend_cache")))

    fs0 = None
    pieces = []
    for idx, (seg, (data, words)) in enumerate(zip(segments, raw)):
        if not data:          # backend returned nothing for this take
            continue
        x, fs = _decode_mp3(data)
        v = VOICES[seg["vkey"]]
        vs = voice_settings_cfg.get(seg["vkey"], {})
        if vs.get("gain_db", v.get("gain_db", 0.0)):
            x = x * (10.0 ** (vs.get("gain_db", v.get("gain_db", 0.0)) / 20.0))
        if fs0 is None:
            fs0 = fs
        em = EMOTIONS.get(seg["emo"], EMOTIONS["neutral"])
        is_filler = bool(seg.get("is_filler", False))
        x, lead = _trim(x, fs)
        words = [(t - lead, d, tx) for (t, d, tx) in words]
        if is_filler:
            g_db = float(seg.get("filler_gain_db",
                                 filler_cfg.get("gain_db", -10.0)))
            x = x * (10.0 ** (g_db / 20.0))
            max_n = int(fs * float(seg.get("filler_max_s", 0.6)))
            if len(x) > max_n:
                x = x[:max_n]
                fn = min(len(x), max(1, int(fs * 0.03)))
                x[-fn:] *= np.linspace(1.0, 0.0, fn)
        elif seg.get("arc_volume_db"):
            x = x * (10.0 ** (float(seg["arc_volume_db"]) / 20.0))

        # ---- Variable word elongation (stretching vowels while thinking) --
        elong_cfg = cfg.get("word_elongation", {})
        if not is_filler and elong_cfg.get("enabled", False):
            x, words = _elongate_words(x, fs, words, rng, elong_cfg)

        pres_hz = persona_cfg.get("presence_hz", 7500.0)
        pres_gain = persona_cfg.get("presence_gain", 0.8)
        pres_q = persona_cfg.get("presence_q", 0.6)

        hp_freq = persona.highpass_hz

        sections = [("hp", hp_freq),
                    ("pk", persona.warm_hz,
                     persona.warm_gain + seg.get("warm", em["warm"]), 0.8),
                    ("pk", persona.pres_hz, persona.pres_gain + seg.get("bright", em["bright"]), 1.1),
                    ("pk", pres_hz, pres_gain, pres_q)]

        for eq in vs.get("eq", []):
            sections.append((eq.get("type", "pk"), eq.get("freq", eq.get("f0", 3600.0)),
                             eq.get("gain", eq.get("gain_db", 1.0)), eq.get("q", 1.0)))
        if em["dark"]:
            sections.append(("lp", em["dark"]))
        x = _eq(x, fs, sections).astype(np.float64)

        if seg["emph"] and emphasis_cfg.get("enabled", True):
            boost_db = emphasis_cfg.get("boost_db", 2.5)
            ramp_factor = emphasis_cfg.get("ramp_factor", 0.35)
            x = _emphasis_env(x, fs, words, seg["emph"], boost_db=boost_db, ramp_factor=ramp_factor)
            # ---- Emphasis brightness (Phase C): accent = brighter too ----
            emph_bc = emphasis_cfg.get("brightness", {}) or {}
            if emph_bc.get("enabled", False):
                x = _emphasis_tilt(x, fs, words, seg["emph"], emph_bc)

        # ---- Vocal effort: gentle projection on content words -------------
        kw_cfg = cfg.get("keyword_emphasis", {})
        if (not is_filler and kw_cfg.get("enabled", False)
                and (seg.get("key_words") or seg.get("low_words"))):
            x = _keyword_emphasis(x, fs, words, rng, seg["key_words"], kw_cfg,
                                  low_words=seg.get("low_words", []))

        # ---- Given-information de-accenting (Phase B) --------------------
        if not is_filler and seg.get("given_words"):
            x = _de_accent(x, fs, words, rng, seg["given_words"], da_cfg)

        if word_life_cfg.get("enabled", False):
            x = _word_life(x, fs, words, rng, db=word_life_cfg.get("db", 1.7))

        # ---- Soft indrawn breath before an emphasised word ---------------
        kbc = cfg.get("keyword_breath", {})
        if (not is_filler and kbc.get("enabled", False)
                and seg.get("key_words")
                and breaths and breath_cfg.get("enabled", False)
                and rng.random() < float(kbc.get("probability", 0.0))):
            ref = float(np.sqrt((x ** 2).mean()) + 1e-12)
            x = _breath_before_keyword(x, fs, words, rng, ref,
                                       seg["emo"], breath_cfg, seg["key_words"])

        if phrase_swell_cfg.get("enabled", False):
            x = _phrase_swell(x, fs, rng, db=phrase_swell_cfg.get("db", 2.6))

        # ---- Natural intonation arc (pitch contour) ----------------------
        pc_cfg   = cfg.get("pitch_contour", {})
        hop_ms   = pc_cfg.get("hop_ms", 20.0)
        hop_smp  = max(1, int(fs * hop_ms / 1000.0))
        n_frames = max(4, (len(x) - hop_smp * 4) // hop_smp)
        contour  = _build_emotion_contour(n_frames, seg["emo"], seg["kind"], cfg)
        x        = _apply_pitch_arc(x.astype(np.float64), fs, contour, hop_ms=hop_ms)

        # ---- Micro-prosody shimmer (natural vocal variation) -------------
        mp_cfg = cfg.get("micro_prosody", {})
        if mp_cfg.get("enabled", False):
            x = _apply_micro_prosody(x.astype(np.float32), fs, rng, mp_cfg)

        # ---- Vocal fry — probabilistic, strongest when pitch is lowered --
        fry_cfg = cfg.get("vocal_fry", {})
        if (not is_filler and fry_cfg.get("enabled", False)
                and rng.random() < float(fry_cfg.get("probability", 1.0))):
            lowered = seg["pitch"] <= persona.base_pitch
            x = _apply_vocal_fry(x.astype(np.float32), fs, rng, fry_cfg,
                                 strength=1.25 if lowered else 0.6)

        if breaths and breath_cfg.get("enabled", False) and not is_filler:
            p = {
                "sentence": breath_cfg.get("sentence_prob", 0.65),
                "question": breath_cfg.get("question_prob", 0.70),
                "exclaim": breath_cfg.get("exclaim_prob", 0.75),
                "ellipsis": breath_cfg.get("ellipsis_prob", 0.70),
                "clause": breath_cfg.get("clause_prob", 0.30),
            }.get(seg["kind"], breath_cfg.get("default_prob", 0.40))
            if idx == 0:
                p = 0.80
            # Adaptive breathing interval: emphatic / warm / substantial
            # passages breathe more often than flat or very short ones.
            p *= float(seg.get("breath_scale", 1.0))
            if len(seg["text"]) < breath_cfg.get("short_phrase_chars", 45):
                p *= float(breath_cfg.get("short_phrase_prob_scale", 1.0))
            p = float(np.clip(p, 0.0, 0.95))
            if seg["emo"] in ("whisper", "tired", "gentle", "sad"):
                p = min(1.0, p + breath_cfg.get("emotional_boost", 0.15))
            size = (0.8 + 0.6 * min(len(seg["text"]) / 70.0, 1.0)) \
                * float(seg.get("breath_scale", 1.0))
            # Thought-gap breathing (Phase E): a longer planned pause needs a
            # fuller breath, so breath size tracks the gap that follows.
            ep = breath_cfg.get("size_from_pause", {}) or {}
            if ep.get("enabled", False):
                ref = float(ep.get("ref_pause_ms", 150.0)) or 150.0
                size *= float(np.clip(seg.get("pause_ms", ref) / ref,
                                      float(ep.get("min_scale", 0.8)),
                                      float(ep.get("max_scale", 1.4))))
            if rng.random() < p:
                ref_rms = float(np.sqrt((x ** 2).mean()) + 1e-12)
                br = _breath(fs, rng, ref_rms, seg["emo"], config=breath_cfg, size=size)
                if br is not None:
                    x = np.concatenate([br, x])
            if seg["kind"] == "ellipsis" and seg["emo"] in (
                    "sad", "tired", "calm", "gentle", "thoughtful",
                    "reassuring") and rng.random() < 0.60:
                ref_rms = float(np.sqrt((x ** 2).mean()) + 1e-12)
                eh = _breath(fs, rng, ref_rms, seg["emo"], config=breath_cfg, exhale=True)
                if eh is not None:
                    x = np.concatenate([x, eh])

        # ---- Mouth click just before speaking (lip parting) --------------
        click_cfg = cfg.get("mouth_click", {})
        if (not is_filler and click_cfg.get("enabled", False)
                and rng.random() < float(click_cfg.get("probability", 0.0))):
            ref = float(np.sqrt((x ** 2).mean()) + 1e-12)
            cl = _mouth_click(fs, rng,
                              duration_ms=float(click_cfg.get("duration_ms", 6.0)))
            cl *= ref * (10.0 ** (float(click_cfg.get("level_db", -38.0)) / 20.0))
            x = np.concatenate([cl.astype(x.dtype), x])

        pieces.append((x.astype(np.float32), seg["pause_ms"]))

    # If the backend returned nothing for any take (degenerate input such as a
    # bare ellipsis), fall back to a valid rate so we still emit a well-formed
    # silent file instead of raising.
    fs = fs0 if fs0 is not None else int(
        cfg.get("output", {}).get("fallback_sample_rate", 24000))
    human = cfg.get("humanize", {})
    lead = int(fs * human.get("lead_ms", 40) / 1000.0)
    tail = int(fs * human.get("tail_ms", 120) / 1000.0)
    overlap = int(fs * human.get("crossfade_ms", 28) / 1000.0)
    total = lead + tail + sum(len(p) + int(fs * max(g, 0) / 1000.0)
                              for p, g in pieces) + fs
    track = np.zeros(total, dtype=np.float64)
    pos = lead
    for i, (x, gap_ms) in enumerate(pieces):
        xf = _fade(x.astype(np.float64).copy(), fs, 4.0)
        if i == 0:
            track[pos:pos + len(xf)] += xf
            pos += len(xf)
            continue
        gap = int(fs * max(gap_ms, 0) / 1000.0)
        if gap <= overlap:
            pos = max(lead, pos - overlap)
            fade = np.linspace(0.0, 1.0, min(overlap, len(xf)))
            n = len(fade)
            track[pos:pos + n] *= (1.0 - fade)
            xf[:n] *= fade
            track[pos:pos + len(xf)] += xf
            pos += len(xf)
        else:
            pos += gap
            track[pos:pos + len(xf)] += xf
            pos += len(xf)
    pos += tail
    track = track[:max(pos, 1)]

    # Shrink the backend's own over-long pauses inside takes down to the
    # configured micro-pause range (they bypass pause_ms entirely).
    track = _compress_gaps(track.astype(np.float64), fs, cfg)

    master_key = steps[0][1].value if isinstance(steps[0][1], Emotion) else steps[0][1]
    master_emotion = EMOTIONS.get(master_key, EMOTIONS["neutral"])
    y, fs_out = master(track, fs, master_emotion, persona, rng, config=cfg,
                       emotion_key=master_key)

    out_wav = os.path.abspath(out_wav)
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    sf.write(out_wav, y, fs_out, subtype="PCM_16")
    return out_wav


def speak(text, emotion=DEFAULT_EMOTION, accent=DEFAULT_ACCENT):
    safe = re.sub(r"\W+", "_", emotion)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", f"say_{safe}_{accent}.wav")
    render(text, out, emotion=emotion, accent=accent)
    try:
        os.startfile(out)
    except Exception:
        pass
    return out


def _find_config_path(config_path: Optional[str] = None) -> str:
    if config_path is not None:
        return config_path
    try:
        from importlib.resources import files
        return str(files("snfy").joinpath("voice_config.yaml"))
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        pass
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(module_dir, "voice_config.yaml")
    if os.path.exists(candidate):
        return candidate
    return os.path.join("configs", "voice_config.yaml")


class SonifyEngine:
    """
    Main voice engine for the Sonify 1026 voice.
    Provides human-like speech synthesis with emotion and accent control.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = _find_config_path(config_path)
        self.config = self._load_config(self.config_path)
        self.persona = Persona()
        self.sample_rate = SR_OUT
        self._volume = 1.0
        self._speed = 1.0
        self._pitch = 0.0
        self._offline_mode = False
        self._offline_engine = None
        self._apply_config()
        logger.info("Sonify 1026 Voice Engine initialized")

    def _apply_config(self) -> None:
        if not self.config:
            return
        character = self.config.get("character", {})
        if character.get("base_pitch") is not None:
            self.persona.base_pitch = character["base_pitch"]

        output = self.config.get("output", {})
        if output.get("sample_rate") is not None:
            self.sample_rate = output["sample_rate"]

        persona_cfg = self.config.get("persona", {})
        for key in ("highpass_hz", "warm_hz", "warm_gain", "pres_hz", "pres_gain",
                     "target_rms_db", "peak_ceiling_db"):
            if persona_cfg.get(key) is not None:
                setattr(self.persona, key, persona_cfg[key])

        global SENTENCE_PAUSE_MS, REGISTERS, DEFAULT_REGISTER
        punct = self.config.get("punctuation", {})
        if punct.get("sentence_pause_ms") is not None:
            SENTENCE_PAUSE_MS = punct["sentence_pause_ms"]

        global EMOTIONS, VOICES, ACCENTS, DEFAULT_ACCENT, VOICE_MODE
        if self.config.get("emotions"):
            EMOTIONS = self.config["emotions"]
        if self.config.get("voices"):
            VOICES = self.config["voices"]
        if self.config.get("voice_mode"):
            VOICE_MODE = {**VOICE_MODE, **self.config["voice_mode"]}
        if self.config.get("accents"):
            ACCENTS = self.config["accents"]
        if self.config.get("default_accent"):
            DEFAULT_ACCENT = self.config["default_accent"]
        if self.config.get("registers"):
            REGISTERS = self.config["registers"]
        if self.config.get("default_register"):
            DEFAULT_REGISTER = self.config["default_register"]

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reload_config(self, config_path: Optional[str] = None) -> None:
        if config_path is not None:
            self.config_path = config_path
        self.config = self._load_config(self.config_path)
        self._apply_config()
        logger.info("Config reloaded")

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.5, min(2.0, speed))

    def set_pitch(self, pitch: float) -> None:
        self._pitch = max(-20.0, min(25.0, pitch))

    def set_offline_mode(self, enabled: bool = True) -> None:
        self._offline_mode = enabled

    def _get_offline_engine(self):
        if self._offline_engine is None:
            import pyttsx3
            self._offline_engine = pyttsx3.init()
        return self._offline_engine

    def _synthesize_offline(self, text: str, rate: float = 1.0, volume: float = 0.9) -> np.ndarray:
        engine = self._get_offline_engine()
        engine.setProperty("rate", int(165 * rate * self._speed))
        engine.setProperty("volume", volume * self._volume)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            audio, _ = sf.read(tmp_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _stream_segments(self, text: str, emotion: str, accent: str):
        import edge_tts
        rng = np.random.default_rng()
        acc = ACCENTS.get(accent, ACCENTS[DEFAULT_ACCENT])
        segs = _segment(text, emotion, self.persona, acc["rate"], rng,
                        config=self.config)
        for seg in segs:
            voice = VOICES[_pick_voice(seg["text"], accent)]["id"]
            try:
                com = edge_tts.Communicate(
                    seg["text"], voice,
                    rate=seg["rate"], volume=seg["volume"], pitch=f"{seg['pitch']:+d}Hz",
                    boundary="WordBoundary",
                )
                audio = bytearray()
                async for ch in com.stream():
                    if ch["type"] == "audio":
                        audio += ch["data"]
                if audio:
                    x, fs = _decode_mp3(bytes(audio))
                    yield x, fs, seg.get("pause_ms", 0)
            except Exception as e:
                logger.warning(f"edge-tts failed for segment, using offline: {e}")
                x = self._synthesize_offline(seg["text"], rate=1.0, volume=0.9)
                yield x, 22050, seg.get("pause_ms", 0)

    def stream(self, text: str, emotion: Emotion = Emotion.NEUTRAL, accent: str = "in"):
        emotion_key = emotion.value if isinstance(emotion, Emotion) else emotion
        return self._stream_segments(text, emotion_key, accent)

    def synthesize(
        self,
        text: str,
        emotion: Emotion = Emotion.NEUTRAL,
        language: Optional[Language] = None,
        output_path: Optional[str] = None,
        accent: str = "in",
        register: str = "normal",
        seed: Optional[int] = None,
        verbatim: Optional[bool] = None,
    ) -> SynthesisResult:
        if language is None:
            language = Language.HI_IN if _is_hindi(text) else Language.EN_IN

        emotion_key = emotion.value if isinstance(emotion, Emotion) else emotion
        if language == Language.EN_US:
            accent_key = "us"
        else:
            accent_key = accent or "in"

        # Phase G — verbatim: read the text exactly as written (skip number /
        # currency / contraction normalisation).  Defaults to the config flag.
        if verbatim is None:
            verbatim = bool((self.config.get("verbatim", {}) or {}).get("enabled", False))

        if output_path is None:
            output_path = os.path.join("outputs", "temp_synthesis.wav")

        cfg = dict(self.config or {})
        cfg["_speed"] = self._speed
        cfg["_pitch"] = self._pitch
        cfg["_volume"] = self._volume

        # Adaptive delivery: each call gets its own performance unless the
        # caller pins a seed for reproducibility.
        if seed is None:
            char = cfg.get("character", {})
            if char.get("randomize_seed", True):
                seed = int(np.random.default_rng().integers(1, 2 ** 31 - 1))
            else:
                seed = int(char.get("seed", 20261009))

        if self._offline_mode:
            audio = self._synthesize_offline(text)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            sf.write(output_path, audio, 22050, subtype="PCM_16")
        else:
            # Pitch guard: verify the rendered voice matches the intended
            # register and correct it if a phrase drifted outside it.
            guard = dict(cfg.get("pitch_guard", {}) or {})
            attempts = int(guard.get("max_attempts", 0)) if guard.get("enabled") else 0
            f_lo = float(guard.get("min_f0_hz", 0.0))
            f_hi = float(guard.get("max_f0_hz", 1e9))
            mid = 0.5 * (f_lo + f_hi)
            max_adj = float(guard.get("max_adjust_hz", 12.0))
            sample_s = float(guard.get("sample_seconds", 6.0))

            adjust = 0.0
            audio = None
            sr = self.sample_rate
            for attempt in range(max(0, attempts) + 1):
                attempt_cfg = dict(cfg)
                attempt_cfg["_pitch"] = self._pitch + adjust
                render(text, output_path, emotion=emotion_key, persona=self.persona,
                       accent=accent_key, seed=seed, config=attempt_cfg,
                       register=register, verbatim=verbatim)
                audio, sr = sf.read(output_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if attempt >= attempts:
                    break
                f0 = _estimate_f0(audio, sr, sample_seconds=sample_s)
                if f0 <= 0.0 or f_lo <= f0 <= f_hi:
                    break
                step = float(np.clip(mid - f0, -max_adj, max_adj))
                if abs(step) < 0.5:
                    break
                adjust += step

        audio, sr = sf.read(output_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sr

        return SynthesisResult(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            emotion=emotion,
            language=language,
            text=text,
        )

    def speak(
        self,
        text: str,
        emotion: Emotion = Emotion.NEUTRAL,
        language: Optional[Language] = None,
    ):
        result = self.synthesize(text, emotion, language)
        try:
            import sounddevice as sd
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except ImportError:
            logger.warning("sounddevice not installed, cannot play audio")
        return result

    def batch_synthesize(
        self,
        texts: List[str],
        emotions: Optional[List[Emotion]] = None,
        output_dir: str = "outputs",
    ) -> List[SynthesisResult]:
        os.makedirs(output_dir, exist_ok=True)
        results = []

        if emotions is None:
            emotions = [Emotion.NEUTRAL] * len(texts)

        for i, (text, emotion) in enumerate(zip(texts, emotions)):
            output_path = os.path.join(output_dir, f"sonify1026_{i:04d}.wav")
            result = self.synthesize(text, emotion, output_path=output_path)
            results.append(result)
            logger.info(f"Synthesized {i+1}/{len(texts)}: {text[:50]}...")

        return results


def create_engine(config_path: Optional[str] = None) -> SonifyEngine:
    return SonifyEngine(config_path)


# Backwards-compatible alias for the previous class name.
BIOS02VoiceEngine = SonifyEngine
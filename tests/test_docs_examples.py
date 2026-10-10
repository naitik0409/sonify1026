"""Verify every code example in README.md and docs/ actually runs."""
import sys
import numpy as np
import soundfile as sf
import snfy

failures = []


def check(name, fn):
    try:
        fn()
        print("  PASS  %s" % name)
    except Exception as e:
        failures.append((name, e))
        print("  FAIL  %s -> %s: %s" % (name, type(e).__name__, e))


print("README / docs example verification")

# --- import forms -------------------------------------------------------
def imports():
    import snfy as a
    from snfy import SonifyEngine, Emotion as b
    from snfy.engine import SonifyEngine as c
    assert a is not None and b is not None and c is not None

check("three import forms", imports)

# --- engine construction forms -----------------------------------------
def engines():
    assert snfy.SonifyEngine() is not None
    assert snfy.create_engine() is not None
    assert snfy.BIOS02VoiceEngine is snfy.SonifyEngine

check("engine construction + legacy alias", engines)

# --- documented exports all exist ---------------------------------------
def exports():
    for n in snfy.__all__:
        assert hasattr(snfy, n), "missing export: %s" % n

check("all __all__ exports resolve", exports)

# --- SynthesisResult fields ---------------------------------------------
def result_fields():
    r = snfy.SonifyEngine().synthesize("Checking result fields.")
    for f in ("audio", "sample_rate", "duration", "emotion", "language", "text"):
        assert hasattr(r, f), "missing field: %s" % f
    assert isinstance(r.audio, np.ndarray) and r.audio.dtype == np.float32
    assert r.sample_rate == 48000

check("SynthesisResult shape", result_fields)

# --- emotion / language enums -------------------------------------------
def enums():
    emotions = [e.value for e in snfy.Emotion]
    assert len(emotions) == 17, "expected 17 emotions, got %d" % len(emotions)
    doc = ["neutral", "happy", "sad", "angry", "excited", "calm", "whisper",
           "fearful", "warm", "gentle", "thoughtful", "playful", "tired",
           "proud", "surprised", "reassuring", "serious"]
    missing = [d for d in doc if d not in emotions]
    assert not missing, "README lists emotions that don't exist: %s" % missing
    for l in (snfy.Language.EN_IN, snfy.Language.HI_IN, snfy.Language.EN_US):
        assert l is not None

check("17 emotions + 3 languages match docs", enums)

# --- README quickstart --------------------------------------------------
def quickstart():
    engine = snfy.SonifyEngine()
    engine.synthesize("That is wonderful news, I am so happy for you!",
                      emotion=snfy.Emotion.HAPPY, output_path="outputs/_doc_quick.wav")
    r = engine.synthesize("Hello again!")
    sf.write("outputs/_doc_mem.wav", r.audio, r.sample_rate)
    engine.synthesize("Nested path works too.", output_path="outputs/_doc/a/b/c.wav")

check("README quickstart", quickstart)

# --- code-switching ------------------------------------------------------
def switching():
    e = snfy.SonifyEngine()
    e.synthesize("Yaar, kal ki meeting bahut important hai.", output_path="outputs/_doc_hing.wav")
    e.synthesize("ठीक है, the point is simple — जो करो, पूरे मन से करो।",
                 output_path="outputs/_doc_mix.wav")
    e.synthesize("The order of 250 units ships on 12/05/2026.",
                 output_path="outputs/_doc_num.wav")

check("Hinglish / script-mix / numbers", switching)

# --- emotion strings -----------------------------------------------------
def emotion_str():
    e = snfy.SonifyEngine()
    e.synthesize("Plain string emotion.", emotion="happy", output_path="outputs/_doc_es.wav")

check("emotion as plain string", emotion_str)

# --- batch ---------------------------------------------------------------
def batch():
    e = snfy.SonifyEngine()
    res = e.batch_synthesize(["Good morning.", "How are you?", "See you soon."],
                             [snfy.Emotion.WARM, snfy.Emotion.CALM, snfy.Emotion.HAPPY],
                             output_dir="outputs/_doc_batch")
    assert len(res) == 3
    assert isinstance(res[0].duration, float)

check("batch_synthesize", batch)

# --- stream --------------------------------------------------------------
def stream():
    import asyncio
    e = snfy.SonifyEngine()

    async def go():
        n = 0
        async for item in e.stream("A longer sentence for streaming."):
            assert isinstance(item, tuple) and len(item) == 3
            audio, sr, pause_ms = item
            assert isinstance(audio, np.ndarray) and isinstance(sr, int)
            assert isinstance(pause_ms, float)
            n += 1
        assert n >= 1
    asyncio.run(go())

check("stream yields (audio, sr, pause_ms)", stream)

# --- runtime controls ----------------------------------------------------
def controls():
    e = snfy.SonifyEngine()
    e.set_speed(1.15); e.set_pitch(0); e.set_volume(1.0)
    e.reload_config()
    r = e.synthesize("Controls applied.", output_path="outputs/_doc_ctl.wav")
    assert r.duration > 0.3

check("runtime controls", controls)

# --- registers -----------------------------------------------------------
def registers():
    e = snfy.SonifyEngine()
    for reg in ("normal", "formal", "casual"):
        e.synthesize(f"Register {reg}.", register=reg, output_path="outputs/_doc_%s.wav" % reg)

check("all three registers", registers)

# --- low-level functions -------------------------------------------------
def lowlevel():
    cfg = snfy.create_engine().config
    snfy.render("Hello.", "outputs/_doc_render.wav", config=cfg)
    snfy.render_arc([("First part.", "neutral"), ("Second part.", "happy")],
                    "outputs/_doc_arc.wav", config=cfg)

check("render() and render_arc()", lowlevel)

# --- logging claim -------------------------------------------------------
def logging_claim():
    import logging
    logging.getLogger("sonify1026").setLevel(logging.WARNING)

check("logger name 'sonify1026'", logging_claim)

# --- config copy recipe from docs ---------------------------------------
def config_recipe():
    import pathlib, shutil
    src = pathlib.Path(snfy.__file__).parent / "voice_config.yaml"
    assert src.exists(), "packaged config not found at %s" % src
    shutil.copy(src, "outputs/_doc_my_voice.yaml")
    e = snfy.create_engine(config_path="outputs/_doc_my_voice.yaml")
    r = e.synthesize("Custom config path.", output_path="outputs/_doc_custom.wav")
    assert r.duration > 0.3

check("config_path recipe + shipped config exists", config_recipe)

# --- reload_config with new path ----------------------------------------
def reload_new_path():
    e = snfy.SonifyEngine()
    e.reload_config("outputs/_doc_my_voice.yaml")
    e.synthesize("Reloaded path.", output_path="outputs/_doc_reload.wav")

check("reload_config(new_path)", reload_new_path)

print()
if failures:
    print("%d example(s) FAILED" % len(failures))
    sys.exit(1)
print("All documentation examples verified.")
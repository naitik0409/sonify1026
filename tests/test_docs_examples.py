"""Verify every documented example and API claim actually works.

Documentation that lies is worse than no documentation, so each example in
README.md and docs/ is executed here.
"""
import numpy as np
import pytest
import soundfile as sf
import snfy


def test_import_forms():
    """README: three equivalent import styles."""
    import snfy as a
    from snfy import SonifyEngine, Emotion as b
    from snfy.engine import SonifyEngine as c
    assert a is not None and b is not None and c is not None


def test_all_exports_resolve():
    """README: every name listed in __all__ is importable."""
    for n in snfy.__all__:
        assert hasattr(snfy, n), "missing export: %s" % n


def test_engine_construction_forms():
    """README: class, factory, and legacy alias."""
    assert snfy.SonifyEngine() is not None
    assert snfy.create_engine() is not None
    assert snfy.BIOS02VoiceEngine is snfy.SonifyEngine


def test_emotion_and_language_enums_match_docs():
    """README tables must match the real enums."""
    emotions = [e.value for e in snfy.Emotion]
    documented = ["neutral", "happy", "sad", "angry", "excited", "calm",
                  "whisper", "fearful", "warm", "gentle", "thoughtful",
                  "playful", "tired", "proud", "surprised", "reassuring",
                  "serious"]
    assert len(emotions) == 17, "expected 17 emotions, got %d" % len(emotions)
    missing = [d for d in documented if d not in emotions]
    assert not missing, "README lists non-existent emotions: %s" % missing
    assert [l.value for l in snfy.Language] == ["en-IN", "hi-IN", "en-US"]


def test_synthesis_result_shape():
    """docs/API.md: SynthesisResult fields and types."""
    r = snfy.SonifyEngine().synthesize("Checking result fields.")
    for f in ("audio", "sample_rate", "duration", "emotion", "language", "text"):
        assert hasattr(r, f), "missing field: %s" % f
    assert isinstance(r.audio, np.ndarray)
    assert r.audio.dtype == np.float32
    assert r.sample_rate == 48000


def test_quickstart_file_and_memory():
    """README quickstart."""
    engine = snfy.SonifyEngine()
    engine.synthesize("That is wonderful news, I am so happy for you!",
                      emotion=snfy.Emotion.HAPPY,
                      output_path="outputs/_doc_quick.wav")
    r = engine.synthesize("Hello again!")
    sf.write("outputs/_doc_mem.wav", r.audio, r.sample_rate)
    assert r.duration > 0.3


def test_nested_output_dir_is_created(tmp_path):
    """README: 'Directories are created automatically'."""
    target = tmp_path / "a" / "b" / "c" / "deep.wav"
    snfy.SonifyEngine().synthesize("Nested path.", output_path=str(target))
    assert target.exists()


def test_code_switching():
    """README: Hinglish, script mix, and number normalisation."""
    e = snfy.SonifyEngine()
    for i, t in enumerate([
        "Yaar, kal ki meeting bahut important hai.",
        "ठीक है, the point is simple — जो करो, पूरे मन से करो।",
        "The order of 250 units ships on 12/05/2026.",
    ]):
        r = e.synthesize(t, output_path="outputs/_doc_switch_%d.wav" % i)
        assert r.duration > 0.5


def test_emotion_accepts_plain_string():
    """README: 'emotion="happy" works'."""
    r = snfy.SonifyEngine().synthesize("Plain string emotion.", emotion="happy",
                                       output_path="outputs/_doc_es.wav")
    assert r.duration > 0.3


def test_accents_and_registers():
    """README: accent and register arguments."""
    e = snfy.SonifyEngine()
    for accent in ("in", "us"):
        e.synthesize("Accent check.", accent=accent,
                     output_path="outputs/_doc_acc_%s.wav" % accent)
    for reg in ("normal", "formal", "casual"):
        e.synthesize("Register check.", register=reg,
                     output_path="outputs/_doc_reg_%s.wav" % reg)


def test_batch_synthesize():
    """README: batch example."""
    res = snfy.SonifyEngine().batch_synthesize(
        ["Good morning.", "How are you?", "See you soon."],
        [snfy.Emotion.WARM, snfy.Emotion.CALM, snfy.Emotion.HAPPY],
        output_dir="outputs/_doc_batch")
    assert len(res) == 3
    assert all(isinstance(r.duration, float) for r in res)


def test_stream_yields_triples():
    """docs/API.md: stream yields (audio, sample_rate, pause_ms)."""
    import asyncio

    async def go():
        n = 0
        async for audio, sr, pause_ms in snfy.SonifyEngine().stream(
                "A longer sentence for streaming."):
            assert isinstance(audio, np.ndarray)
            assert isinstance(sr, int)
            assert isinstance(pause_ms, float)
            n += 1
        assert n >= 1

    asyncio.run(go())


def test_runtime_controls_and_reload():
    """README: set_speed / set_pitch / set_volume / reload_config."""
    e = snfy.SonifyEngine()
    e.set_speed(1.15)
    e.set_pitch(0)
    e.set_volume(1.0)
    e.reload_config()
    assert e.synthesize("Controls applied.",
                        output_path="outputs/_doc_ctl.wav").duration > 0.3


def test_low_level_functions():
    """docs/API.md: render() and render_arc()."""
    cfg = snfy.create_engine().config
    snfy.render("Hello.", "outputs/_doc_render.wav", config=cfg)
    snfy.render_arc([("First part.", "neutral"), ("Second part.", "happy")],
                    "outputs/_doc_arc.wav", config=cfg)


def test_config_path_and_reload(tmp_path):
    """docs/CONFIGURATION.md: custom config recipe."""
    import pathlib
    import shutil
    src = pathlib.Path(snfy.__file__).parent / "voice_config.yaml"
    assert src.exists(), "packaged config missing at %s" % src
    dst = tmp_path / "my_voice.yaml"
    shutil.copy(src, dst)

    e = snfy.create_engine(config_path=str(dst))
    assert e.synthesize("Custom config.", output_path=str(tmp_path / "c.wav")).duration > 0.3
    e.reload_config(str(dst))
    assert e.synthesize("Reloaded.", output_path=str(tmp_path / "d.wav")).duration > 0.3


def test_documented_config_keys_exist():
    """docs/CONFIGURATION.md: every section named in the guide is real."""
    cfg = snfy.create_engine().config
    for section in ["character", "output", "emotions", "voices", "accents",
                    "voice_mode",
                    "persona", "punctuation", "humanize", "pitch_guard",
                    "prosody", "dynamic_prosody", "breath", "keyword_emphasis",
                    "keyword_breath", "micro_prosody", "vocal_fry",
                    "mouth_click", "pause_filler", "word_life", "phrase_swell",
                    "registers", "variation", "voice_settings",
                    "sentence_type_rules", "pitch_contour", "word_elongation",
                    "room_tone", "air_breath", "gap_compress", "emotion_tags"]:
        assert section in cfg, "documented section missing from config: %s" % section


def test_logger_name():
    """Troubleshooting: logger is 'sonify1026'."""
    import logging
    logging.getLogger("sonify1026").setLevel(logging.WARNING)
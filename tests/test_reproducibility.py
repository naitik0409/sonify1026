"""Reproducibility tests.

The seed pins the engine's performance decisions, but the upstream Edge TTS
service is not bit-deterministic, so a fixed seed alone is not byte-identical.
The optional `backend_cache` reuses the raw neural audio for identical takes,
which makes a fixed seed fully reproducible. These tests verify that cache
behaviour without depending on the network.
"""
import asyncio

import numpy as np

import snfy
import snfy.engine as E


def _seg(text="Hello there.", pitch=0):
    return {"text": text, "vkey": "neerja", "rate": "+0%", "volume": "+0%",
            "pitch": pitch}


def test_cache_reuses_audio(monkeypatch):
    calls = {"n": 0}

    async def fake(seg, voice):
        calls["n"] += 1
        return b"\x01\x02\x03", [(0.0, 0.1, "x")]

    monkeypatch.setattr(E, "_edge_one", fake)
    E.clear_backend_cache()
    cfg = {"enabled": True, "max_entries": 8}
    asyncio.run(E._edge_all([_seg()], "neerja", cache_cfg=cfg))
    asyncio.run(E._edge_all([_seg()], "neerja", cache_cfg=cfg))
    assert calls["n"] == 1


def test_cache_off_refetches(monkeypatch):
    calls = {"n": 0}

    async def fake(seg, voice):
        calls["n"] += 1
        return b"", []

    monkeypatch.setattr(E, "_edge_one", fake)
    E.clear_backend_cache()
    cfg = {"enabled": False}
    asyncio.run(E._edge_all([_seg()], "neerja", cache_cfg=cfg))
    asyncio.run(E._edge_all([_seg()], "neerja", cache_cfg=cfg))
    assert calls["n"] == 2


def test_cache_key_includes_parameters(monkeypatch):
    calls = {"n": 0}

    async def fake(seg, voice):
        calls["n"] += 1
        return b"", []

    monkeypatch.setattr(E, "_edge_one", fake)
    E.clear_backend_cache()
    cfg = {"enabled": True, "max_entries": 8}
    asyncio.run(E._edge_all([_seg(pitch=0)], "neerja", cache_cfg=cfg))
    asyncio.run(E._edge_all([_seg(pitch=5)], "neerja", cache_cfg=cfg))
    assert calls["n"] == 2


def test_cache_evicts_oldest(monkeypatch):
    async def fake(seg, voice):
        return b"", []

    monkeypatch.setattr(E, "_edge_one", fake)
    E.clear_backend_cache()
    cfg = {"enabled": True, "max_entries": 1}
    asyncio.run(E._edge_all([_seg("one")], "neerja", cache_cfg=cfg))
    asyncio.run(E._edge_all([_seg("two")], "neerja", cache_cfg=cfg))
    assert len(E._EDGE_AUDIO_CACHE) <= 1


def test_reproducible_with_cache(tmp_path):
    import copy
    import hashlib

    eng = snfy.SonifyEngine()
    c = copy.deepcopy(eng.config)
    c["backend_cache"] = {"enabled": True, "max_entries": 16}
    eng.config = c
    txt = "That is wonderful news, I am so happy for you!"
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    eng.synthesize(txt, emotion="happy", seed=12345, output_path=str(a))
    eng.synthesize(txt, emotion="happy", seed=12345, output_path=str(b))
    assert hashlib.sha256(a.read_bytes()).digest() == \
        hashlib.sha256(b.read_bytes()).digest()

"""Codec motion-vector conversion tests independent of PyAV installation."""
import io
import json

import numpy as np
import pytest

from mtr_kv.config import Config
from mtr_kv.video import motion_vector_block
from mtr_kv.video import run_video


def test_motion_vector_block_uses_destination_and_fractional_scale():
    vector = {"w": 16, "h": 16, "dst_x": 40, "dst_y": 24,
              "motion_x": 12, "motion_y": -4, "motion_scale": 4}
    assert motion_vector_block(vector, 100, 100) == {
        "rect": [32, 16, 16, 16], "mv": [3.0, -1.0], "confidence": 1.0,
    }


def test_motion_vector_block_clips_to_frame():
    vector = {"w": 16, "h": 16, "dst_x": 3, "dst_y": 3,
              "motion_x": 0, "motion_y": 0, "motion_scale": 4}
    assert motion_vector_block(vector, 100, 100)["rect"] == [0, 0, 11, 11]


def test_video_decoder_feeds_online_controller(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "moving.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width = 128
        stream.height = 128
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            image[32:64, 20 + 8 * index:52 + 8 * index] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    sink = io.StringIO()
    run_video(Config(), str(path), sink)
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(records) == 4
    assert all(record["type"] == "frame_result" for record in records)
    assert all(record["status"] == "READY" for record in records)
    assert any(record["motion_vectors"] > 0 for record in records[1:])
    assert all(record["decode_seconds"] >= 0 for record in records)
    assert all(record["end_to_end_frame_seconds"] >= record["frame_processing_seconds"] for record in records)
    semantic_sink = io.StringIO()
    run_video(Config(), str(path), semantic_sink, max_frames=2, semantic_model=True)
    semantic_records = [json.loads(line) for line in semantic_sink.getvalue().splitlines()]
    assert len(semantic_records) == 2
    assert all(record["semantic_feedback"]["status"] == "APPLIED" for record in semantic_records)
    assert all(record["semantic_model"] for record in semantic_records)
    assert all(record["tensor_storage"]["payload_bytes"] > 0 for record in semantic_records)
    assert semantic_records[0]["memory"]["physical_bytes"] < 1_000_000
    stored_sink = io.StringIO()
    run_video(Config(), str(path), stored_sink, max_frames=2, semantic_model=True,
              kv_inference=True, compare_baseline=True)
    stored_records = [json.loads(line) for line in stored_sink.getvalue().splitlines()]
    assert len(stored_records) == 2
    assert all(record["inference_path"] == "stored_kv" for record in stored_records)
    assert all(record["kv_consumed_pages"] > 0 for record in stored_records)
    assert all(record["tensor_storage"]["bytes_read"] > 0 for record in stored_records)
    assert all(record["baseline_comparison"]["same_top1"] for record in stored_records)


def test_video_cross_frame_reuse_with_real_model_and_decoder(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    import timm

    # Exercise the full model offline with reproducible local random weights.
    create_model = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, **kw: create_model(name, **{**kw, "pretrained": False}))
    path = tmp_path / "static.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=12)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0", "g": "120", "bf": "0", "sc_threshold": "0"}
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[30:75, 25:70] = 200
        for _ in range(12):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    sink = io.StringIO()
    run_video(Config(), str(path), sink, semantic_model=True, kv_inference=True,
              kv_reuse=True, compare_baseline=True)
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(records) == 12
    assert all(r["status"] == "READY" for r in records)
    assert records[0]["kv_reuse"]["pages_reused"] == 0
    assert any(r["kv_reuse"]["pages_reused"] > 0 for r in records)
    assert all(r["baseline_comparison"]["same_top1"] for r in records)
    from mtr_kv.evaluate import summarize
    summary = summarize(records)
    assert summary["kv_pages_reused"] > 0
    assert summary["kv_tokens_reused"] > 0
    assert summary["peak_reuse_validation_input_bytes"] > 0

"""Evidence runs are paired, hashed, and clearly labeled exploratory."""
import json

import pytest

from mtr_kv.evidence import capture


@pytest.mark.parametrize("enable_reuse", [False, True])
def test_capture_alternates_arms_and_records_provenance(tmp_path, monkeypatch, enable_reuse):
    video = tmp_path / "real-input.mp4"
    video.write_bytes(b"provided video bytes")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    calls = []
    reuse_calls = []

    def fake_run(_cfg, _source, sink, max_frames, semantic_model, kv_inference,
                 compare_baseline=False, device="cpu", kv_reuse=False, reuse_window=8):
        calls.append((kv_inference, compare_baseline, device))
        reuse_calls.append(kv_reuse)
        for generation in range(1, max_frames + 1):
            record = {
                "type": "frame_result", "generation": generation, "source_pts": generation * 10,
                "status": "READY", "motion_vectors": 3, "metadata_fallback": False,
                "inference_path": "stored_kv" if kv_inference else "standard",
                "model_inference_seconds": 0.01 if kv_inference else 0.02,
                "frame_processing_seconds": 0.03, "end_to_end_frame_seconds": 0.04,
                "tensor_storage": {"payload_bytes": 128, "bytes_written": 128, "bytes_read": 128,
                                   "device": device},
            }
            if compare_baseline:
                record["baseline_comparison"] = {
                    "baseline_seconds": 0.02, "same_top1": True,
                    "mean_abs_logit_error": 0.001,
                }
            if kv_reuse:
                record["kv_reuse"] = {"pages_reused": 1, "pages_recomputed": 0,
                                      "tokens_kv_reused": 32, "validation_input_bytes": 128}
            sink.write(json.dumps(record) + "\n")

    monkeypatch.setattr("mtr_kv.evidence.run_video", fake_run)
    monkeypatch.setattr("mtr_kv.evidence._runtime", lambda: {"python": "test"})
    monkeypatch.setattr("mtr_kv.evidence._revision", lambda: {"head": "abc", "dirty": False})
    output = tmp_path / "evidence"
    manifest_path = capture(str(video), str(output), str(config), repeats=2,
                            max_frames=3, warmup_frames=1, kv_reuse=enable_reuse)
    manifest = json.loads(manifest_path.read_text())
    assert calls == [(False, False, "cpu"), (True, False, "cpu"),
                     (True, False, "cpu"), (False, False, "cpu"), (True, True, "cpu")]
    assert manifest["status"] == "complete"
    assert reuse_calls == [False, enable_reuse, enable_reuse, False, enable_reuse]
    assert manifest["kv_reuse"]["enabled"] == enable_reuse
    assert manifest["quality_run"]["summary"]["kv_pages_reused"] == (2 if enable_reuse else 0)
    assert manifest["median_model_ms_by_arm"] == {"standard": 20.0, "tiered": 10.0}
    assert manifest["quality_run"]["summary"]["top1_agreement_rate"] == 1.0
    assert manifest["source"]["sha256"] and (output / "config.json").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Paired classifier check" in report
    assert "patentability opinion" in report
    with pytest.raises(FileExistsError):
        capture(str(video), str(output), str(config), repeats=1, max_frames=3, warmup_frames=1)


def test_capture_rejects_missing_input_and_bad_warmup(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="existing local"):
        capture(str(tmp_path / "missing.mp4"), str(tmp_path / "out"), str(config))
    video = tmp_path / "input.mp4"
    video.write_bytes(b"content")
    with pytest.raises(ValueError, match="warmup_frames"):
        capture(str(video), str(tmp_path / "out"), str(config), max_frames=3, warmup_frames=3)


def test_capture_preserves_incomplete_manifest_on_runtime_failure(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"content")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("mtr_kv.evidence._runtime", lambda: (_ for _ in ()).throw(RuntimeError("runtime unavailable")))
    output = tmp_path / "failed-evidence"
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        capture(str(video), str(output), str(config), max_frames=2, warmup_frames=0)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert "runtime unavailable" in (output / "report.md").read_text()


def test_capture_runs_real_decoder_model_and_tensor_path(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    video = tmp_path / "codec-fixture.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width = 128
        stream.height = 128
        stream.pix_fmt = "yuv420p"
        for index in range(2):
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            image[20:55, 20 + index * 5:55 + index * 5] = 200
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    manifest_path = capture(str(video), str(tmp_path / "evidence"), str(config),
                            repeats=1, max_frames=2, warmup_frames=0)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["quality_run"]["summary"]["compared_frames"] == 2
    assert manifest["runs"][1]["summary"]["mean_kv_consumed_pages"] > 0

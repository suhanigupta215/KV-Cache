"""Measured-run evaluation stays separate from simulator estimates."""
import json

import pytest

from mtr_kv.evaluate import evaluate_file, summarize


def test_summarize_measured_frames():
    frame = {
        "type": "frame_result", "status": "READY", "metadata_fallback": False,
        "motion_vectors": 12, "inference_path": "stored_kv",
        "model_inference_seconds": 0.02, "frame_processing_seconds": 0.03,
        "kv_consumed_pages": 7,
        "tensor_storage": {"payload_bytes": 1024, "bytes_written": 2048, "bytes_read": 1024},
        "baseline_comparison": {"baseline_seconds": 0.01, "same_top1": True,
                                "mean_abs_logit_error": 0.001},
    }
    result = summarize([frame, {**frame, "metadata_fallback": True}])
    assert result["frames"] == 2
    assert result["metadata_fallback_rate"] == 0.5
    assert result["top1_agreement_rate"] == 1
    assert result["mean_kv_consumed_pages"] == 7
    assert result["peak_actual_tensor_payload_bytes"] == 1024


def test_evaluate_file_rejects_invalid_or_empty_input(tmp_path):
    source = tmp_path / "run.jsonl"
    source.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        evaluate_file(str(source), str(tmp_path / "out.json"))
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no frame_result"):
        evaluate_file(str(source), str(tmp_path / "out.json"))
    source.write_text(json.dumps({"type": "frame_result"}) + "\n", encoding="utf-8")
    output = evaluate_file(str(source), str(tmp_path / "out.json"))
    assert json.loads(output.read_text())["frames"] == 1

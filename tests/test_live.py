"""Online controller contract for real producer-supplied observations."""
import io
import json

from mtr_kv.config import Config
from mtr_kv.live import LiveController, run_stream


def test_live_stream_is_incremental_and_rejects_stale_feedback():
    source = io.StringIO("\n".join([
        json.dumps({"type": "frame", "generation": 1, "blocks": []}),
        json.dumps({"type": "semantic", "generation": 0, "page_scores": {"0": 1}}),
        json.dumps({"type": "frame", "generation": 2, "blocks": []}),
    ]) + "\n")
    sink = io.StringIO()
    run_stream(Config(), (640, 480), source, sink)
    results = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [result["status"] for result in results] == ["READY", "STALE_DROP", "READY"]
    assert results[0]["metadata_fallback"] is True
    assert results[0]["ready_tokens"][0]["generation"] == 1


def test_live_semantic_feedback_promotes_current_quiet_page():
    cfg = Config()
    cfg.motion.score_mode = "motion_only"
    cfg.temporal.persistence_frames = 1
    cfg.temporal.stable_frames = 1
    controller = LiveController(cfg, (640, 480))
    frame = controller.frame({
        "generation": 1, "frame_type": "P",
        "blocks": [{"rect": [0, 0, 640, 480], "mv": [0, 0], "confidence": 1}],
    })
    assert frame["status"] == "READY"
    assert frame["memory"]["hot"] == 0
    correction = controller.semantic({"generation": 1, "page_scores": {"0": 1.0}})
    assert correction["apply_statuses"] == ["APPLIED"]
    assert correction["memory"]["hot"] == 1


def test_semantic_corrections_do_not_reuse_future_frame_descriptor_ids():
    cfg = Config()
    cfg.motion.score_mode = "motion_only"
    cfg.temporal.persistence_frames = cfg.temporal.stable_frames = 1
    controller = LiveController(cfg, (640, 480))
    ids = set()
    for generation in range(1, 8):
        result = controller.frame({
            "generation": generation, "frame_type": "P",
            "blocks": [{"rect": [0, 0, 640, 480], "mv": [0, 0], "confidence": 1}],
        })
        assert result["status"] == "READY"
        for descriptor in result["descriptors"]:
            assert descriptor["descriptor_id"] not in ids
            ids.add(descriptor["descriptor_id"])
        correction = controller.semantic({
            "generation": generation, "page_scores": {str(page): 1.0 for page in range(controller.page_count)},
        })
        assert correction["apply_statuses"] == ["APPLIED"] * controller.page_count
        assert correction["memory"]["hot"] == controller.page_count
        assert not controller.kmc.invariants()


def test_live_bad_record_does_not_stop_stream():
    source = io.StringIO('{bad json\n{"type":"frame","generation":1,"blocks":[]}\n')
    sink = io.StringIO()
    run_stream(Config(), (640, 480), source, sink)
    results = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [result["type"] for result in results] == ["error", "frame_result"]

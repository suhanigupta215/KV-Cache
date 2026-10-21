"""The demonstration uses a real encoded source and measured decisions."""
import io
import json

import pytest

from mtr_kv.config import Config
from mtr_kv.demo import create_demo_video
from mtr_kv.video import run_video


def test_generated_demo_decodes_and_changes_residency(tmp_path):
    pytest.importorskip("av")
    path = create_demo_video(str(tmp_path / "demo.mp4"), fps=12, seconds=4)
    sink = io.StringIO()
    previews = []
    run_video(Config(), str(path), sink,
              on_frame=lambda record, frame: previews.append((record["generation"], frame.width)))
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(records) == len(previews) == 48
    assert records[0]["metadata_fallback"]
    assert all(record["status"] == "READY" for record in records)
    assert any(record["motion_vectors"] > 0 for record in records)
    assert len({tuple(page["tier"] for page in record["pages"]) for record in records}) > 1
    assert all(page["ready"] for record in records for page in record["pages"])
    assert records[-1]["source_seconds"] > records[0]["source_seconds"]
    with pytest.raises(ValueError, match="already exists"):
        create_demo_video(str(path))


def test_demo_rejects_invalid_duration(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        create_demo_video(str(tmp_path / "invalid.mp4"), seconds=0)

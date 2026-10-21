"""Phone transport tests use a real live encoded pipe, without phone hardware."""
import io
import json
from pathlib import Path
import sys
import textwrap
import time

import pytest

from mtr_kv.config import Config
from mtr_kv.phone import PhoneOptions, PhoneVideo, connected_devices, require_tool, scrcpy_command
from mtr_kv.video import run_video


def test_phone_options_and_command(monkeypatch):
    monkeypatch.setattr("mtr_kv.phone.require_tool", lambda name: "/tools/" + name)
    options = PhoneOptions("ABC-123", "front", 640, 15)
    assert PhoneOptions.from_uri(options.uri()) == options
    command = scrcpy_command(options, "/tmp/private-camera.mkv")
    assert "--serial=ABC-123" in command
    assert "--no-audio" in command
    assert "--video-source=camera" in command
    assert "--video-codec=h264" in command
    assert "--record-format=mkv" in command
    assert "--select-usb" in scrcpy_command(PhoneOptions(), "/tmp/camera.mkv")


@pytest.mark.parametrize("options", [{"serial": "x; touch /tmp/bad"}, {"fps": 0}, {"fps": 61},
                                         {"size": 641}, {"facing": "screen"}])
def test_invalid_phone_options(options):
    with pytest.raises(ValueError):
        PhoneOptions(**options)


def test_missing_tools_are_actionable(monkeypatch):
    monkeypatch.setattr("mtr_kv.phone.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="brew install"):
        require_tool("scrcpy")


def test_device_listing_reports_authorization(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("mtr_kv.phone.require_tool", lambda name: name)
    monkeypatch.setattr("mtr_kv.phone.subprocess.run", lambda *a, **kw: SimpleNamespace(
        stdout="List of devices attached\nABC unauthorized usb:1\nDEF device usb:2 model:Example\n"))
    devices = connected_devices()
    assert [d["status"] for d in devices] == ["unauthorized", "device"]
    assert all(d["usb"] for d in devices)


@pytest.fixture
def fake_phone(monkeypatch, tmp_path):
    pytest.importorskip("av")
    executable = tmp_path / "scrcpy-test"
    executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''
        import sys, time
        import av
        import numpy as np
        path = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--record="))
        with av.open(path, "w", format="matroska", options={"flush_packets": "1", "cluster_time_limit": "50"}) as out:
            stream = out.add_stream("libx264", rate=30)
            stream.width = stream.height = 128
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "30", "bf": "0"}
            for index in range(900):
                image = np.zeros((128, 128, 3), dtype=np.uint8)
                x = (index * 3) % 96
                image[32:64, x:x+32] = 240
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    out.mux(packet)
                time.sleep(.02)
    '''))
    executable.chmod(0o700)
    monkeypatch.setattr("mtr_kv.phone.require_tool", lambda name: str(executable) if name == "scrcpy" else "/usr/bin/true")


def test_phone_live_pipe_decodes_vectors_and_cleans_up(fake_phone):
    sink = io.StringIO()
    run_video(Config(), PhoneOptions(size=640).uri(), sink, max_frames=8,
              on_frame=lambda *_: time.sleep(.08))
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(records) == 8
    assert all(r["status"] == "READY" for r in records)
    assert all(r["source_kind"] == "android_usb_camera" for r in records)
    assert any(r["motion_vectors"] > 0 for r in records)
    assert records[-1]["capture_dropped_frames"] > 0


def test_phone_stop_releases_process_and_fifo(fake_phone):
    with PhoneVideo(PhoneOptions()) as capture:
        assert next(capture.decode(capture.streams.video[0])).width == 128
        process = capture.process
        directory = Path(capture.temp.name)
    assert process.poll() is not None
    assert not capture.worker.is_alive()
    assert not directory.exists()


def test_phone_startup_failure_is_actionable(monkeypatch):
    monkeypatch.setattr("mtr_kv.phone.require_tool", lambda _: "/usr/bin/false")
    with pytest.raises(RuntimeError, match="USB debugging"):
        with PhoneVideo(PhoneOptions()):
            pass

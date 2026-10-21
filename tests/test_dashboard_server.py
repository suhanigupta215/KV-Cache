"""Local dashboard serves live records without exposing source credentials."""
import json
import base64
import io
from threading import Thread
from threading import Event
import time
from urllib.request import urlopen

from mtr_kv.dashboard import create_server
from mtr_kv.demo import create_demo_video
import pytest


def test_local_dashboard_stream_and_config(monkeypatch):
    def fake_video(_config, _source, sink, **_options):
        sink.write(json.dumps({"type": "frame_result", "generation": 1}) + "\n")
        sink.flush()

    monkeypatch.setattr("mtr_kv.dashboard.run_video", fake_video)
    server = create_server("https://user:secret@example.org/clip.mp4?token=private", port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/config") as response:
            config = json.load(response)
        assert config["source"] == "https://example.org/clip.mp4"
        with urlopen(base + "/api/stream") as response:
            stream = response.read().decode("utf-8")
        assert '"generation": 1' in stream
        assert "event: done" in stream
        with urlopen(base + "/") as response:
            assert b"Live controller trace" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_phone_camera_disconnect_releases_exclusive_session(monkeypatch):
    finished = Event()

    def fake_video(_config, _source, sink, **options):
        try:
            sink.write('{"type":"frame_result","generation":1}\n')
            sink.flush()
            deadline = time.monotonic() + 5
            while not options["should_stop"]() and time.monotonic() < deadline:
                time.sleep(.02)
        finally:
            finished.set()

    monkeypatch.setattr("mtr_kv.dashboard.run_video", fake_video)
    server = create_server("phone://usb?facing=back&size=640&fps=15", port=0, realtime=False)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/config") as response:
            config = json.load(response)
        assert config["source_kind"] == "android_usb_camera"
        assert config["source"] == "Android USB · back camera"
        response = urlopen(base + "/api/stream", timeout=5)
        assert b'"generation":1' in response.readline()
        try:
            with urlopen(base + "/api/stream", timeout=5) as duplicate:
                assert b"already in use" in duplicate.read()
        finally:
            response.close()
        assert finished.wait(2), "camera must stop when its browser connection closes"
        assert server.phone_lock.acquire(timeout=1)
        server.phone_lock.release()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_dashboard_streams_synchronized_decoded_preview(tmp_path):
    pytest.importorskip("av")
    from PIL import Image

    source = create_demo_video(str(tmp_path / "demo.mp4"), seconds=1)
    server = create_server(str(source), port=0, max_frames=2, realtime=False)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/stream", timeout=20) as response:
            events = response.read().decode("utf-8").split("\n\n")
        records = [json.loads(event.removeprefix("data: ")) for event in events if event.startswith("data: ")]
        assert [record["generation"] for record in records] == [1, 2]
        for record in records:
            image = Image.open(io.BytesIO(base64.b64decode(record["preview_jpeg"])))
            assert image.size == (640, 360)
            assert record["pages"]
            assert record["status"] == "READY"
        assert any("event: done" in event for event in events)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

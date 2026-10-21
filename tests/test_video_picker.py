"""Exercise real HTTP parsing, validation and video decoding without client sockets."""
import io
import json
import os
import tempfile
import unittest
from http.client import HTTPResponse
from pathlib import Path
from unittest.mock import patch
from mtr_kv.dashboard import DashboardHandler, create_server, MAX_UPLOAD_BYTES
from mtr_kv.demo import create_demo_video

class MemorySocket:
    def __init__(self, data):
        self.input = io.BytesIO(data)
        self.output = io.BytesIO()
        self.timeout = None
    def makefile(self, mode, *args): return self.input
    def sendall(self, data): self.output.write(data)
    def settimeout(self, timeout): self.timeout = timeout
    def gettimeout(self): return self.timeout

class VideoPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp")
        cls.video = create_demo_video(str(Path(cls.inputs.name) / "test.mp4"), seconds=1).read_bytes()
    @classmethod
    def tearDownClass(cls): cls.inputs.cleanup()
    def setUp(self):
        self.environment = patch.dict(os.environ, {"MTR_KV_UPLOAD_ROOT": self.inputs.name})
        self.environment.start()
        self.server = create_server("original.mp4", port=0, max_frames=2, realtime=False)
    def tearDown(self):
        self.server.server_close()
        self.environment.stop()
    def request(self, method, path, data=b"", headers=None):
        headers = {"Host": "127.0.0.1:8765", "Content-Length": str(len(data)), **(headers or {})}
        raw = f"{method} {path} HTTP/1.0\r\n" + "".join(f"{key}: {value}\r\n" for key, value in headers.items()) + "\r\n"
        connection = MemorySocket(raw.encode() + data)
        with patch("mtr_kv.dashboard.select.select", return_value=([], [], [])), patch.object(DashboardHandler, "log_message"):
            DashboardHandler(connection, ("127.0.0.1", 12345), self.server,
                             directory=str(Path(__file__).resolve().parents[1] / "frontend"))
        response = HTTPResponse(MemorySocket(connection.output.getvalue()))
        response.begin()
        return response.status, response.read()
    def upload(self, data=None, name="test.mp4", **headers):
        status, body = self.request("POST", "/api/videos", self.video if data is None else data,
                                    {"Content-Type": "application/octet-stream", "X-Video-Name": name, **headers})
        return status, json.loads(body)
    def test_selected_video_runs_actual_decoder_and_can_replay(self):
        status, uploaded = self.upload(name="gallery%20clip.mp4", Origin="http://127.0.0.1:8765")
        self.assertEqual(status, 201)
        self.assertEqual(uploaded["source"], "gallery clip.mp4")
        self.assertEqual(uploaded["codec"], "h264")
        self.assertEqual((uploaded["width"], uploaded["height"]), (640, 360))
        for _ in range(2):
            status, body = self.request("GET", "/api/stream?video=" + uploaded["video_id"])
            self.assertEqual(status, 200)
            events = body.decode().split("\n\n")
            records = [json.loads(e[6:]) for e in events if e.startswith("data: ")]
            self.assertEqual([r["generation"] for r in records], [1, 2])
            self.assertTrue(all(r["status"] == "READY" and r["pages"] and r["preview_jpeg"] for r in records))
            self.assertTrue(any("event: done" in e for e in events))
    def test_uploads_are_isolated_and_do_not_replace_the_default(self):
        one = self.upload(name="one.mp4")[1]
        two = self.upload(name="two.mp4")[1]
        sources = []
        def capture(_cfg, source, sink, **options): sources.append(source)
        with patch("mtr_kv.dashboard.run_video", capture):
            for token in [one["video_id"], two["video_id"], one["video_id"]]:
                self.request("GET", "/api/stream?video=" + token)
            self.request("GET", "/api/stream")
        self.assertEqual(sources[0], sources[2])
        self.assertNotEqual(sources[0], sources[1])
        self.assertEqual(sources[3], "original.mp4")
    def test_rejects_corrupt_and_empty_files_without_retaining_them(self):
        self.assertEqual(self.upload(b"this is not video")[0], 400)
        self.assertEqual(self.upload(b"")[0], 400)
        self.assertEqual(list(Path(self.server.upload_directory.name).iterdir()), [])
    def test_rejects_cross_origin_uploads_and_html_forms(self):
        self.assertEqual(self.upload(Origin="http://unrelated.example")[0], 403)
        self.assertEqual(self.upload(**{"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.upload(**{"Content-Type": "text/plain"})[0], 415)
    def test_rejects_large_upload_before_reading_the_body(self):
        status, result = self.upload(b"x", **{"Content-Length": str(MAX_UPLOAD_BYTES + 1)})
        self.assertEqual(status, 413)
        self.assertIn("512 MB", result["error"])
    def test_unknown_video_id_does_not_start_default_video(self):
        self.assertEqual(self.request("GET", "/api/stream?video=not-an-upload")[0], 404)
    def test_filename_cannot_escape_temporary_storage(self):
        status, result = self.upload(name="..%2F..%2Fgallery.mp4")
        self.assertEqual(status, 201)
        self.assertEqual(result["source"], "gallery.mp4")
        temporary_path = Path(self.server.uploads[result["video_id"]]["path"])
        self.assertEqual(temporary_path.parent, Path(self.server.upload_directory.name))
        self.assertEqual(self.request("GET", "/" + temporary_path.name)[0], 404)
    def test_closing_server_removes_uploaded_video(self):
        self.upload()
        directory = Path(self.server.upload_directory.name)
        self.server.server_close()
        self.assertFalse(directory.exists())
    def test_config_advertises_picker_and_existing_run_mode(self):
        status, body = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        config = json.loads(body)
        self.assertTrue(config["video_upload"])
        self.assertFalse(config["semantic_model"])
        self.assertEqual(config["max_upload_bytes"], MAX_UPLOAD_BYTES)

if __name__ == "__main__": unittest.main()

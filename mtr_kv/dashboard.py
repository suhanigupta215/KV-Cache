"""Local dashboard for decoded frame decisions and browser-selected videos."""
from __future__ import annotations

from functools import partial
import base64
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import time
import select
import socket
import secrets
import tempfile
from threading import Event, Lock
from urllib.parse import urlsplit, parse_qs, unquote

from .config import load_config
from .video import run_video


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_STORED_BYTES = 2 * 1024 * 1024 * 1024


def source_label(source: str) -> str:
    if source.startswith("phone://"):
        from .phone import PhoneOptions

        return f"Android USB · {PhoneOptions.from_uri(source).facing} camera"
    parsed = urlsplit(source)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.hostname}{parsed.path}"
    return Path(source).name


class StreamSink:
    def __init__(self, output):
        self.output = output

    def write(self, record: str) -> None:
        for line in record.splitlines():
            if line:
                self.output.write(f"data: {line}\n\n".encode("utf-8"))

    def flush(self) -> None:
        self.output.flush()


class DashboardServer(ThreadingHTTPServer):
    def server_close(self):
        if hasattr(self, "stop_capture"):
            self.stop_capture.set()
            # Give the request's capture context time to terminate scrcpy.
            if self.phone_lock.acquire(timeout=10):
                self.phone_lock.release()
        super().server_close()
        if hasattr(self, "upload_directory"):
            self.upload_directory.cleanup()


class DashboardHandler(SimpleHTTPRequestHandler):
    def json_response(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/api/videos":
            self.json_response(404, {"error": "Unknown endpoint."})
            return
        # A raw binary request cannot be sent by a cross-origin HTML form.
        origin = self.headers.get("Origin")
        if ((origin and urlsplit(origin).netloc != self.headers.get("Host"))
                or self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self.json_response(403, {"error": "Open the picker on this dashboard to select a video."})
            return
        if self.headers.get_content_type() != "application/octet-stream":
            self.json_response(415, {"error": "Select a video using the dashboard video picker."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if size <= 0 or self.headers.get("Transfer-Encoding"):
            self.json_response(400, {"error": "The video is empty or its file size is missing."})
            return
        if size > MAX_UPLOAD_BYTES:
            self.json_response(413, {"error": "Choose a video smaller than 512 MB."})
            return
        name = unquote(self.headers.get("X-Video-Name", "video.mp4"))
        name = Path(name.replace("\\", "/")).name[:200] or "video.mp4"
        token = secrets.token_hex(16)
        # Never use the submitted name as a filesystem path.
        path = Path(self.server.upload_directory.name) / (token + ".video")
        with self.server.upload_lock:
            if self.server.upload_bytes + size > MAX_STORED_BYTES:
                self.json_response(413, {"error": "Temporary video storage is full. Restart the dashboard to clear it."})
                return
            self.server.upload_bytes += size
        registered = False
        timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(30)
            remaining = size
            with path.open("xb") as output:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("The upload was interrupted. Please select the video again.")
                    output.write(chunk)
                    remaining -= len(chunk)
            import av
            try:
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise ValueError("This file does not contain a video stream.")
                    stream = container.streams.video[0]
                    first = next(container.decode(stream), None)
                    if first is None:
                        raise ValueError("This video contains no decodable frames.")
                    info = {"codec": stream.codec_context.name,
                            "width": first.width, "height": first.height}
            except ValueError:
                raise
            except Exception as error:
                raise ValueError("This video cannot be decoded. Try an H.264 MP4 video.") from error
            with self.server.upload_lock:
                self.server.uploads[token] = {"path": str(path), "name": name}
            registered = True
            self.json_response(201, {"video_id": token, "source": name, "size_bytes": size, **info})
        except ImportError:
            self.json_response(503, {"error": "Video decoding is unavailable. Install requirements-live.txt and restart the server."})
        except (ValueError, OSError) as error:
            self.json_response(400, {"error": str(error)})
        finally:
            self.connection.settimeout(timeout)
            if not registered:
                path.unlink(missing_ok=True)
                with self.server.upload_lock:
                    self.server.upload_bytes -= size

    def do_GET(self) -> None:
        if self.path == "/api/config":
            payload = json.dumps({"source": source_label(self.server.video_source),
                                  "semantic_model": self.server.semantic_model,
                                  "kv_inference": self.server.kv_inference,
                                  "kv_reuse": self.server.kv_reuse,
                                  "video_upload": True,
                                  "max_upload_bytes": MAX_UPLOAD_BYTES,
                                  "source_kind": "android_usb_camera" if self.server.video_source.startswith("phone://") else "video",
                                  "compare_baseline": self.server.compare_baseline}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if urlsplit(self.path).path == "/api/stream":
            query = parse_qs(urlsplit(self.path).query)
            video_id = query.get("video", [None])[0]
            source = self.server.video_source
            if video_id is not None:
                with self.server.upload_lock:
                    uploaded = self.server.uploads.get(video_id)
                if not uploaded:
                    self.json_response(404, {"error": "The selected video is no longer available. Select it again."})
                    return
                source = uploaded["path"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sink = StreamSink(self.wfile)
            phone = source.startswith("phone://")
            owns_camera = False
            started = time.monotonic()
            first_pts = None

            def disconnected():
                if self.server.stop_capture.is_set():
                    return True
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable:
                    try:
                        return self.connection.recv(1, socket.MSG_PEEK) == b""
                    except OSError:
                        return True
                return False

            def preview(record, frame):
                nonlocal first_pts, started
                pts = record.get("source_seconds")
                if pts is not None:
                    if first_pts is None:
                        first_pts = pts
                        # Model initialization must not make playback rush to catch up.
                        started = time.monotonic()
                    delay = pts - first_pts - (time.monotonic() - started)
                    if self.server.realtime and delay > 0:
                        time.sleep(min(delay, 1.0))
                image = frame.to_image()
                image.thumbnail((640, 360))
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=75)
                record["preview_jpeg"] = base64.b64encode(encoded.getvalue()).decode("ascii")
            try:
                if phone:
                    owns_camera = self.server.phone_lock.acquire(blocking=False)
                    if not owns_camera:
                        raise RuntimeError("The phone camera is already in use by another run. Stop that run before connecting again.")
                run_video(load_config(self.server.config_path), source, sink,
                          max_frames=self.server.max_frames, semantic_model=self.server.semantic_model,
                          kv_inference=self.server.kv_inference,
                          compare_baseline=self.server.compare_baseline, device=self.server.device,
                          on_frame=preview, should_stop=disconnected,
                          kv_reuse=self.server.kv_reuse, reuse_window=self.server.reuse_window)
                self.wfile.write(b"event: done\ndata: {}\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                try:
                    self.wfile.write(f"event: run_error\ndata: {json.dumps({'message': str(error)})}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                if owns_camera:
                    self.server.phone_lock.release()
            return
        if self.path.startswith("/api/"):
            self.send_error(404)
            return
        super().do_GET()


def create_server(source: str, host: str = "127.0.0.1", port: int = 8765,
                  config_path: str = "configs/reference.json", max_frames: int | None = None,
                  semantic_model: bool = False, kv_inference: bool = False,
                  compare_baseline: bool = False, device: str = "cpu",
                  realtime: bool = True, kv_reuse: bool = False, reuse_window: int = 8) -> ThreadingHTTPServer:
    if (kv_inference or compare_baseline) and not semantic_model:
        raise ValueError("KV inference and baseline comparison require --semantic-model")
    if kv_reuse and not (semantic_model and kv_inference):
        raise ValueError("--kv-reuse requires --semantic-model and --kv-inference")
    if reuse_window <= 0:
        raise ValueError("reuse_window must be positive")
    server = DashboardServer((host, port), partial(DashboardHandler, directory=str(FRONTEND)))
    server.video_source = source
    server.config_path = config_path
    server.max_frames = max_frames
    server.semantic_model = semantic_model
    server.kv_inference = kv_inference
    server.compare_baseline = compare_baseline
    server.device = device
    server.kv_reuse = kv_reuse
    server.reuse_window = reuse_window
    server.realtime = realtime
    server.phone_lock = Lock()
    server.stop_capture = Event()
    upload_root = os.environ.get("MTR_KV_UPLOAD_ROOT")
    if upload_root:
        Path(upload_root).mkdir(parents=True, exist_ok=True)
    server.upload_directory = tempfile.TemporaryDirectory(prefix="mtr-kv-videos-", dir=upload_root, ignore_cleanup_errors=True)
    server.upload_lock = Lock()
    server.uploads = {}
    server.upload_bytes = 0
    return server

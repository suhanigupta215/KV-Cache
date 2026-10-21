"""USB Android camera capture through scrcpy, using real encoded H.264.

The private FIFO carries Matroska bytes in memory; no camera recording is
saved. A decoder drains the camera continuously and retains one latest frame
so slow inference cannot build an ever-growing video backlog.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import io
import os
from pathlib import Path
import queue
import re
import select
import shutil
import subprocess
import tempfile
from threading import Event, Thread
import time
from urllib.parse import parse_qs, urlencode, urlsplit


@dataclass(frozen=True)
class PhoneOptions:
    serial: str | None = None
    facing: str = "back"
    size: int = 640
    fps: int = 15

    def __post_init__(self):
        if self.facing not in ("back", "front", "external"):
            raise ValueError("camera-facing must be back, front, or external")
        if not 128 <= self.size <= 3840 or self.size % 2:
            raise ValueError("camera size must be an even number from 128 to 3840")
        if not 1 <= self.fps <= 60:
            raise ValueError("camera fps must be from 1 to 60")
        if self.serial is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.serial):
            raise ValueError("invalid Android device serial")

    def uri(self) -> str:
        values = {"facing": self.facing, "size": self.size, "fps": self.fps}
        if self.serial:
            values["serial"] = self.serial
        return "phone://usb?" + urlencode(values)

    @classmethod
    def from_uri(cls, source: str):
        parsed = urlsplit(source)
        if parsed.scheme != "phone" or parsed.netloc != "usb":
            raise ValueError("phone source must use phone://usb")
        values = parse_qs(parsed.query)
        if set(values) - {"serial", "facing", "size", "fps"}:
            raise ValueError("unknown phone source option")
        return cls(serial=values.get("serial", [None])[0],
                   facing=values.get("facing", ["back"])[0],
                   size=int(values.get("size", [640])[0]), fps=int(values.get("fps", [15])[0]))


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is not installed. On macOS run: brew install scrcpy android-platform-tools")
    return path


def connected_devices() -> list[dict]:
    result = subprocess.run([require_tool("adb"), "devices", "-l"],
                            capture_output=True, text=True, timeout=15, check=True)
    devices = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] in ("device", "unauthorized", "offline"):
            devices.append({"serial": fields[0], "status": fields[1],
                            "usb": any(field.startswith("usb:") for field in fields[2:]),
                            "details": " ".join(fields[2:])})
    return devices


def scrcpy_command(options: PhoneOptions, fifo: str) -> list[str]:
    command = [require_tool("scrcpy"), "--video-source=camera", "--video-codec=h264",
               "--no-audio", "--no-playback", "--no-window",
               f"--camera-facing={options.facing}", f"--max-size={options.size}",
               f"--camera-fps={options.fps}", "--record-format=mkv", f"--record={fifo}"]
    command += [f"--serial={options.serial}"] if options.serial else ["--select-usb"]
    return command


class _PipeReader(io.RawIOBase):
    def __init__(self, path, process, stopped, timeout=20):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.process, self.stopped, self.timeout = process, stopped, timeout

    def readable(self):
        return True

    def read(self, size=-1):
        deadline = time.monotonic() + self.timeout
        while not self.stopped():
            readable, _, _ = select.select([self.fd], [], [], 0.1)
            if readable:
                try:
                    data = os.read(self.fd, size if size > 0 else 32768)
                except BlockingIOError:
                    continue
                if data:
                    return data
                if self.process.poll() is not None:
                    return b""
                # A FIFO with no connected writer is readable as EOF.
                time.sleep(0.02)
            if self.process.poll() is not None:
                return b""
            if time.monotonic() >= deadline:
                raise TimeoutError("Phone camera sent no video for 20 seconds. Check the USB connection and camera availability.")
        return b""

    def close(self):
        if not self.closed:
            os.close(self.fd)
        super().close()


class PhoneVideo:
    """Small input-container adapter compatible with run_video's decode loop."""

    def __init__(self, options: PhoneOptions, should_stop=None):
        self.options = options
        self.should_stop = should_stop or (lambda: False)
        self.stop = Event()
        self.ready = Event()
        self.finished = Event()
        self.latest = queue.Queue(maxsize=1)
        self.logs = deque(maxlen=30)
        self.error = None
        self.dropped_frames = 0
        self.streams = None
        self.process = None
        self.temp = None

    def cancelled(self):
        return self.stop.is_set() or self.should_stop()

    def __enter__(self):
        if self.cancelled():
            raise InterruptedError("Phone camera connection cancelled")
        if not hasattr(os, "mkfifo"):
            raise RuntimeError("USB phone capture currently supports macOS and Linux; use an encoded network stream on Windows")
        require_tool("adb")
        # Validate dependencies before creating resources.
        require_tool("scrcpy")
        self.temp = tempfile.TemporaryDirectory(prefix="mtr-kv-phone-")
        fifo = str(Path(self.temp.name) / "camera.mkv")
        os.mkfifo(fifo, 0o600)
        try:
            self.process = subprocess.Popen(scrcpy_command(self.options, fifo),
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, text=True)
            self.log_worker = Thread(target=self._read_logs, daemon=True)
            self.log_worker.start()
            self.worker = Thread(target=self._decode, args=(fifo,), daemon=True)
            self.worker.start()
            deadline = time.monotonic() + 30
            while not self.ready.wait(0.1):
                if self.cancelled():
                    raise InterruptedError("Phone camera connection cancelled")
                if time.monotonic() > deadline:
                    raise TimeoutError("Phone camera did not start within 30 seconds. Check USB authorization and Android camera support.")
            if self.error:
                raise RuntimeError(self._error_message()) from self.error
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _read_logs(self):
        for line in self.process.stderr:
            self.logs.append(line.strip())

    def _error_message(self):
        details = "\n".join(self.logs)
        return f"Phone camera failed: {self.error}. Check Android 12+, USB debugging, authorization, and camera availability.\n{details}"

    def _decode(self, fifo):
        import av

        try:
            with _PipeReader(fifo, self.process, self.cancelled) as reader:
                with av.open(reader, format="matroska", options={"analyzeduration": "0", "probesize": "32768"}) as container:
                    if not container.streams.video:
                        raise ValueError("phone stream contains no video")
                    stream = container.streams.video[0]
                    stream.codec_context.options = {"flags2": "+export_mvs"}
                    self.streams = container.streams
                    self.ready.set()
                    for frame in container.decode(stream):
                        if self.cancelled():
                            break
                        try:
                            self.latest.put_nowait(frame)
                        except queue.Full:
                            try:
                                self.latest.get_nowait()
                                self.dropped_frames += 1
                            except queue.Empty:
                                pass
                            self.latest.put_nowait(frame)
        except Exception as error:
            if not self.cancelled():
                self.error = error
        finally:
            self.ready.set()
            self.finished.set()

    def decode(self, _stream):
        while not self.cancelled():
            try:
                yield self.latest.get(timeout=0.1)
            except queue.Empty:
                if self.finished.is_set():
                    if self.error:
                        raise RuntimeError(self._error_message()) from self.error
                    raise RuntimeError("Phone camera disconnected. Reconnect the cable and restart the camera.")

    def __exit__(self, *_args):
        self.stop.set()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            if hasattr(self, "worker"):
                self.worker.join(timeout=3)
            if hasattr(self, "log_worker"):
                self.log_worker.join(timeout=1)
            self.process.stderr.close()
        if self.temp:
            self.temp.cleanup()


def open_video_source(source: str, should_stop=None):
    if source.startswith("phone://"):
        return PhoneVideo(PhoneOptions.from_uri(source), should_stop)
    import av

    return av.open(source)

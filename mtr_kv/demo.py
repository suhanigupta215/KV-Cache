"""Generate a repeatable compressed video for a local, offline controller demo."""
from pathlib import Path


def create_demo_video(output: str, fps: int = 12, seconds: int = 18) -> Path:
    """Encode still, moving, still, and scene-cut phases; never invent vectors."""
    import av
    import numpy as np

    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must be positive")
    path = Path(output)
    if path.exists():
        raise ValueError(f"output already exists: {path}; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    rng = np.random.default_rng(42)
    background = rng.integers(16, 48, (height, width, 3), dtype=np.uint8)
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast", "g": str(fps * 6), "bf": "0"}
        for index in range(fps * seconds):
            phase = min(3, index * 4 // (fps * seconds))
            image = background.copy()
            if phase == 3:
                image = 255 - image
            # Motion begins only in phase two; it stops again in phase three.
            x = 64 if phase != 1 else 64 + (index * 9) % 440
            image[140:220, x:x + 80] = [35, 190, 225]
            image[48:108, 480:580] = [230, 170, 60]
            # Phase indicators are part of the actual encoded source.
            for marker in range(4):
                image[310:326, 40 + marker * 145:160 + marker * 145] = (
                    [65, 205, 130] if marker == phase else [75, 80, 90])
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path

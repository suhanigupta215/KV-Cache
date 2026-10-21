# Android phone camera over USB

The `phone` command connects an Android 12+ camera to MTR-KV using scrcpy and
ADB. It requests a real H.264 camera stream over the USB connection. It does
not mirror the screen or require you to open the phone's Camera app. Audio is
disabled. Encoded bytes pass through a private temporary FIFO, without saving
a camera recording. Saved dashboard traces contain measurements rather than
preview images.

## Set up once on macOS

```bash
brew install scrcpy android-platform-tools
```

On your Android phone:

1. Enable Developer options (usually by tapping Build number seven times).
2. Enable **USB debugging** in Developer options.
3. Connect a data-capable USB cable to the Mac, unlock the phone, and accept
   its USB debugging authorization prompt for this computer.
4. Close any app already using the camera.

Check the connection from the project directory:

```bash
.venv/bin/python -m mtr_kv phone --list-devices
```

The selected device should have status `device`, rather than `unauthorized`
or `offline`. The phone must support scrcpy camera capture (Android 12+).
USB debugging provides this computer with development access; disable it or
revoke the computer's authorization on the phone when you no longer need it.

## Start the live dashboard

```bash
.venv/bin/python -m mtr_kv phone
```

Open http://127.0.0.1:8767 and select **Connect camera**. Point the phone at a
scene and move it: the synchronized preview, controller page tiers, motion
vector counts, and fallback events update continuously.

To enable model-backed K/V consumption and the classifier comparison:

```bash
.venv/bin/python -m mtr_kv phone --semantic-model --kv-inference --compare-baseline
```

First use can download model weights. Start the server in codec-only mode if
you want to check the connection without loading the model first.

Add `--kv-reuse` to the model-backed command to reuse unchanged static-page
K/V across processed frames. Exact final-block input validation is mandatory;
see [cross-frame reuse](KV_REUSE.md). Moving scenes can legitimately have zero
cache hits.

Select **Stop** to disconnect the capture. **Connect camera** starts a new
session with a fresh controller. Only one camera run can be active per server;
stop the previous browser session before starting another. Ctrl-C stops the
server; a disconnected or inactive capture also times out and releases its
scrcpy process. Unplugging the phone reports a run error; reconnect and select
**Connect camera** again. Automatic reconnection is not implemented.

## Select a camera or device

```bash
.venv/bin/python -m mtr_kv phone --camera-facing front
.venv/bin/python -m mtr_kv phone --serial YOUR_ADB_SERIAL --size 640 --fps 15
```

The default selects the single USB device, back camera, maximum dimension of
640 pixels, and 15 requested capture frames per second. Supported camera sizes
and rates depend on the phone. Use `scrcpy --list-cameras` and
`scrcpy --list-camera-sizes` for its declared capabilities; if camera startup
fails, try another size or rate. Android versions older than 12 are not
supported by this direct camera adapter. On Windows use an encoded network
stream; the FIFO transport currently targets macOS and Linux.

## Latency and measurements

The camera decoder drains the compressed stream continuously. If inference
falls behind, it retains the most recent decoded frame and skips older frames
for controller/model processing. The dashboard reports the cumulative number
skipped. Controller generations count processed observations, and temporal
persistence therefore applies to those observations rather than every captured
frame. Decoder motion vectors still refer to adjacent encoded reference
frames; they are not motion accumulated over skipped frames.

This bounds the adapter's decoded-frame backlog. Camera hardware encoding,
scrcpy's Matroska mux buffering, USB transport, model loading, and browser
display still add latency. The displayed processing rate is not a measurement
of camera-to-display latency, and no fixed real-time frame rate is guaranteed.
Missing motion metadata takes the same conservative fallback as file input.
Phone-side H.264 K/V data is not involved: the optional ViT model creates K/V
tensors on the computer after video decoding.

## Verification boundary

Automated integration tests send real H.264/Matroska through the FIFO into the
actual PyAV decoder and controller, exercising motion metadata, bounded frame
retention, and capture cleanup. They substitute an encoded test producer for
the physical phone. A hardware trial with your phone is still required to
verify its encoder, supported camera settings, and macOS USB authorization.

References: [scrcpy camera capture](https://github.com/Genymobile/scrcpy/blob/master/doc/camera.md),
[scrcpy recording](https://github.com/Genymobile/scrcpy/blob/master/doc/recording.md),
[Android USB debugging](https://developer.android.com/tools/adb#Enabling).

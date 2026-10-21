# Select a video in the dashboard

Start the local demo from the project root:

```powershell
.\.venv\Scripts\python.exe -m mtr_kv demo
```

Open http://127.0.0.1:8765. In **Analyze your own video**, select **Choose File**, choose a video from your computer or gallery, and select **Analyze video**. The browser transfers it to this local server, validates that it can decode, and starts a new run. Upload progress and errors appear next to the picker.

The decoded preview, page tiers, motion-vector counts, and frame table update while it runs. **Video results** appears at completion. **Stop** shows partial results; **Restart run** replays the selected file. **Save trace** exports the latest 10,000 image-free frame records as JSONL. Summary metrics also cover only this bounded trace when the run is longer.

To include the AI classifier, actual tiered K/V consumption, exact reuse checks, and baseline agreement, start the server with:

```powershell
.\.venv\Scripts\python.exe -m mtr_kv demo --semantic-model --kv-inference --kv-reuse --compare-baseline
```

The picker preserves the analysis mode chosen at server startup. It explicitly displays whether model analysis is enabled. First model use can require a weight download. No model results are fabricated when the server is in codec/controller mode.

H.264 MP4 is recommended. Other formats depend on FFmpeg/PyAV support; absent exported motion vectors produce conservative metadata fallback. There is no guarantee of reuse for a selected video. Classifier agreement measures consistency with standard ViT, not ground-truth accuracy.

Each file is limited to 512 MB. Temporary uploads are stored outside the served frontend, with a total temporary budget of 2 GiB, and cleaned when the server closes. Selected videos use independent IDs, so one tab cannot change another tab's selected source. Restarting the server requires selecting the file again. Use the default localhost binding to keep the service local.

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_video_picker -v
node tests/video_picker_ui.cjs
```

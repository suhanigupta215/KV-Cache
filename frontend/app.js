"use strict";

const EXAMPLE_ARTIFACT = {
  config: { name: "frozen-reference-v1", version: "1.0" },
  scenario: { name: "T03 semantic rescue", seed: 20260922 },
  runs: {
    baseline: { metrics: { pages_avoided: 0, bytes_saved: 0, semantic_rescues: 0, stale_drops: 0, fallback_rate: 0, baseline_pages_total: 80, hot_pages_total: 80, resident_pages_total: 80 }, frames: [] },
    codec_only: { metrics: { pages_avoided: 80, bytes_saved: 377487360, semantic_rescues: 0, stale_drops: 0, fallback_rate: 0, baseline_pages_total: 80, hot_pages_total: 0, resident_pages_total: 80 }, frames: [] },
    full: { metrics: { pages_avoided: 52, bytes_saved: 245366784, semantic_rescues: 20, stale_drops: 0, fallback_rate: 0, baseline_pages_total: 80, hot_pages_total: 28, resident_pages_total: 80 }, frames: [] },
  },
};

let activeArtifact = null;
let activeMode = "full";

function normalizeArtifact(payload) {
  if (!payload || typeof payload !== "object" || !payload.scenario || typeof payload.scenario !== "object") throw new Error("scenario metadata is missing");
  if (!payload.runs || typeof payload.runs !== "object") throw new Error("simulation runs are missing");
  const runs = {};
  for (const [mode, run] of Object.entries(payload.runs)) {
    if (run && typeof run.metrics === "object" && run.metrics !== null) runs[mode] = { metrics: run.metrics, frames: Array.isArray(run.frames) ? run.frames : [] };
  }
  if (Object.keys(runs).length === 0) throw new Error("no run includes metrics");
  return { config: payload.config && typeof payload.config === "object" ? payload.config : {}, scenario: payload.scenario, runs };
}

function availableModes(data) {
  return Object.keys(data.runs);
}

function metricValue(value, suffix = "") {
  if (value === undefined || value === null || (typeof value === "number" && !Number.isFinite(value))) return "—";
  if (typeof value === "number") return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value)}${suffix}`;
  return `${value}${suffix}`;
}

function modeDescription(mode) {
  return {
    baseline: "Every page is fully materialized in the fast T0 tier. This is the conventional comparison point.",
    codec_only: "Codec motion and temporal evidence guide tier placement before transformer semantics are available.",
    full: "Codec guidance is combined with semantic rescue, promoting quiet pages that later prove important.",
    semantic_only: "Semantic evidence controls residency after model computation, without early codec guidance.",
  }[mode] || "This result mode follows the MTR-KV simulator artifact.";
}

function renderDashboard(data, requestedMode) {
  const modes = availableModes(data);
  const mode = modes.includes(requestedMode) ? requestedMode : modes[0];
  const metrics = data.runs[mode].metrics;
  activeArtifact = data;
  activeMode = mode;
  document.getElementById("scenario-name").textContent = data.scenario.name || "Untitled scenario";
  document.getElementById("scenario-meta").textContent = `Seed ${data.scenario.seed ?? "—"} · ${data.config.name || "simulator artifact"} v${data.config.version || "—"}`;
  const select = document.getElementById("mode-select");
  select.replaceChildren(...modes.map((name) => new Option(name.replaceAll("_", " "), name)));
  select.value = mode;
  const cards = [["Pages avoided", metricValue(metrics.pages_avoided), "Lower fast-tier pressure", "metric-cyan"], ["Bytes saved", metricValue(metrics.bytes_saved), "Compared with baseline", "metric-violet"], ["Semantic rescues", metricValue(metrics.semantic_rescues), "Quiet but important pages", "metric-green"], ["Safety checks", metricValue(metrics.stale_drops), "Stale updates rejected", "metric-amber"]];
  const metricGrid = document.getElementById("metric-grid");
  metricGrid.replaceChildren(...cards.map(([label, value, detail, style]) => {
    const card = document.createElement("article"); card.className = `metric ${style}`;
    const labelNode = document.createElement("span"); labelNode.className = "metric-label"; labelNode.textContent = label;
    const valueNode = document.createElement("strong"); valueNode.className = "metric-value"; valueNode.textContent = value;
    const detailNode = document.createElement("span"); detailNode.className = "metric-detail"; detailNode.textContent = detail;
    card.append(labelNode, valueNode, detailNode);
    return card;
  }));
  renderResidencyChart(metrics);
  renderFrameTimeline(data.runs[mode].frames);
  if (data.runs[mode].frames.length === 0) {
    setExplanation(`${mode.replaceAll("_", " ")} mode`, "What this view means", modeDescription(mode), [
      ["Fallback rate", metricValue((metrics.fallback_rate ?? 0) * 100, "%")],
      ["Source", "deterministic simulator output"],
    ]);
  }
}

function renderResidencyChart(metrics) {
  const chart = document.getElementById("residency-chart");
  const rows = [
    ["Baseline pages", metrics.baseline_pages_total],
    ["Fast T0 pages", metrics.hot_pages_total],
    ["Resident pages", metrics.resident_pages_total],
  ];
  const maximum = Math.max(...rows.map(([, value]) => Number(value) || 0), 1);
  chart.replaceChildren(...rows.map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "chart-track";
    const fill = document.createElement("div");
    fill.className = "chart-fill";
    fill.style.width = `${Math.max(0, Math.min(100, ((Number(value) || 0) / maximum) * 100))}%`;
    track.append(fill);
    const amount = document.createElement("span");
    amount.className = "chart-value";
    amount.textContent = metricValue(value);
    row.append(name, track, amount);
    return row;
  }));
}

function eventLabel(frame) {
  if (frame.semantic_rescues > 0) return ["Semantic rescue", "rescue"];
  if (frame.fallback) return ["Fallback", "fallback"];
  if (frame.transition_predicted) return ["Transition", "transition"];
  return ["Tier plan", ""];
}

function renderFrameTimeline(frames) {
  const body = document.getElementById("frame-table-body");
  body.replaceChildren();
  if (frames.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No frame records are available.";
    row.append(cell);
    body.append(row);
    return;
  }
  frames.forEach((frame, index) => {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `Explain frame ${frame.frame_index ?? index}`);
    const [label, style] = eventLabel(frame);
    const cells = [frame.frame_index ?? index, frame.generation ?? "—", frame.pages_avoided ?? "—", `${frame.mtr_pages_t0 ?? 0} / ${frame.mtr_pages_t1 ?? 0} / ${frame.mtr_pages_t2 ?? 0}`];
    cells.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
    const event = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = `event-tag ${style}`;
    tag.textContent = label;
    event.append(tag);
    row.append(event);
    const select = () => selectFrame(frame, row);
    row.addEventListener("click", select);
    row.addEventListener("keydown", (eventKey) => { if (eventKey.key === "Enter" || eventKey.key === " ") { eventKey.preventDefault(); select(); } });
    body.append(row);
  });
  selectFrame(frames[0], body.firstElementChild);
}

function selectFrame(frame, row) {
  document.querySelectorAll("#frame-table-body tr").forEach((item) => item.classList.remove("is-selected"));
  row.classList.add("is-selected");
  const [label] = eventLabel(frame);
  const explanation = frame.semantic_rescues > 0
    ? "Transformer semantics marked a low-motion page as important, so the controller promoted it safely."
    : frame.fallback
      ? "Codec metadata was unavailable or uncertain, so MTR-KV chose conservative fast-tier residency."
      : frame.transition_predicted
        ? "A scene change was predicted, so the temporal engine avoided aggressive demotion."
        : "Codec and temporal evidence produced a normal tiering decision for this frame.";
  setExplanation(`Frame ${frame.frame_index ?? "—"} · ${label}`, "Why this decision was made", explanation, [
    ["Generation", frame.generation ?? "—"], ["Pages avoided", frame.pages_avoided ?? "—"], ["Tier residency", `T0 ${frame.mtr_pages_t0 ?? 0} · T1 ${frame.mtr_pages_t1 ?? 0} · T2 ${frame.mtr_pages_t2 ?? 0}`],
  ]);
}

function setExplanation(kicker, title, message, details) {
  const panel = document.getElementById("frame-explanation");
  panel.replaceChildren();
  const kickerNode = document.createElement("p"); kickerNode.className = "explanation-kicker"; kickerNode.textContent = kicker;
  const titleNode = document.createElement("h3"); titleNode.textContent = title;
  const messageNode = document.createElement("p"); messageNode.textContent = message;
  const list = document.createElement("ul"); list.className = "explanation-list";
  details.forEach(([name, value]) => { const item = document.createElement("li"); const bold = document.createElement("b"); bold.textContent = `${name}: `; item.append(bold, document.createTextNode(value)); list.append(item); });
  panel.append(kickerNode, titleNode, messageNode, list);
}

function setLoadStatus(message, isError = false) {
  const status = document.getElementById("load-status");
  status.textContent = message;
  status.classList.toggle("is-error", isError);
  status.setAttribute("role", isError ? "alert" : "status");
}

function loadArtifactFile(event) {
  const [file] = event.target.files;
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = normalizeArtifact(JSON.parse(reader.result));
      renderDashboard(data, activeMode);
      document.getElementById("simulation-section").hidden = false;
      document.getElementById("simulation-timeline").hidden = false;
      setLoadStatus(`Loaded ${file.name}. Select a mode or frame to explore the trace.`);
    } catch (error) {
      setLoadStatus(`Invalid result artifact: ${error.message}`, true);
    }
  };
  reader.onerror = () => setLoadStatus("Invalid result artifact: the file could not be read", true);
  reader.readAsText(file);
}

renderDashboard(normalizeArtifact(EXAMPLE_ARTIFACT), activeMode);
document.getElementById("mode-select").addEventListener("change", (event) => renderDashboard(activeArtifact, event.target.value));
document.getElementById("artifact-input").addEventListener("change", loadArtifactFile);
document.getElementById("reset-results").addEventListener("click", () => {
  document.getElementById("artifact-input").value = "";
  renderDashboard(normalizeArtifact(EXAMPLE_ARTIFACT), "full");
  document.getElementById("simulation-section").hidden = false;
  document.getElementById("simulation-timeline").hidden = false;
  setLoadStatus("Showing the built-in presentation example.");
});

let liveStream = null;
let liveFrameCount = 0;
let liveFallbackCount = 0;
let liveRecords = [];
let liveStartedAt = 0;
let livePhone = false;
let liveVideoId = null;
let liveSourceName = "";
let liveConfig = null;
let videoUpload = null;

function showVideoSelection(message, isError = false) {
  const status = document.getElementById("video-selection");
  status.textContent = message;
  status.classList.toggle("is-error", isError);
}

function renderRunSummary(label = "Video results") {
  if (!liveRecords.length) return;
  const frames = liveRecords;
  const measured = frames.filter((r) => Number.isFinite(r.end_to_end_frame_seconds));
  const comparisons = frames.filter((r) => typeof r.baseline_comparison?.same_top1 === "boolean");
  const payloads = frames.filter((r) => Number.isFinite(r.tensor_storage?.payload_bytes));
  const reuse = frames.filter((r) => r.kv_reuse);
  const meanMs = measured.length ? measured.reduce((sum, r) => sum + r.end_to_end_frame_seconds, 0) * 1000 / measured.length : null;
  const fallbackCount = frames.filter((r) => r.metadata_fallback).length;
  const agreed = comparisons.filter((r) => r.baseline_comparison.same_top1).length;
  const values = [
    ["Frames in summary", metricValue(frames.length), `${frames.filter((r) => r.status === "READY").length} ready frames`],
    ["Mean frame time", metricValue(meanMs, " ms"), "Decode + processing, excluding preview and pacing"],
    ["Metadata fallbacks", `${fallbackCount} / ${frames.length}`, "Conservative handling of missing motion information"],
    ["Peak tensor payload", payloads.length ? metricValue(Math.max(...payloads.map((r) => r.tensor_storage.payload_bytes)), " B") : "Not enabled", "Stored tensor representations only"],
    ["K/V pages reused", reuse.length ? metricValue(reuse.reduce((sum, r) => sum + r.kv_reuse.pages_reused, 0)) : "Not enabled", "Total reuse hits across summarized frames"],
    ["Classifier agreement", comparisons.length ? `${agreed} / ${comparisons.length}` : "Not enabled", "Same top-1 prediction as standard ViT"],
  ];
  document.getElementById("video-result-metrics").replaceChildren(...values.map(([name, value, detail]) => {
    const card = document.createElement("article"); card.className = "metric metric-cyan";
    for (const [text, className, tag] of [[name, "metric-label", "span"], [value, "metric-value", "strong"], [detail, "metric-detail", "span"]]) {
      const node = document.createElement(tag); node.className = className; node.textContent = text; card.append(node);
    }
    return card;
  }));
  const first = frames[0];
  document.getElementById("video-results-title").textContent = label;
  document.getElementById("video-results-meta").textContent = `${liveSourceName} — ${first.source_codec || "video"}, ${(first.source_size || []).join(" × ")} pixels. ${liveFrameCount > frames.length ? "Summary covers the most recent 10,000 frames. " : ""}Save trace to keep the frame records.`;
  document.getElementById("video-results").hidden = false;
}

document.getElementById("video-input").addEventListener("change", (event) => {
  const file = event.target.files[0];
  const tooLarge = file && file.size > (liveConfig?.max_upload_bytes || 512 * 1024 * 1024);
  document.getElementById("video-analyze").disabled = !file || !file.size || tooLarge;
  showVideoSelection(!file ? "No video selected." : tooLarge ? "Choose a video smaller than 512 MB." : !file.size ? "This file is empty. Choose another video." : `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB). Ready to analyze.`, Boolean(tooLarge || (file && !file.size)));
});

document.getElementById("video-analyze").addEventListener("click", () => {
  const file = document.getElementById("video-input").files[0];
  if (!file || videoUpload) return;
  if (liveStream) liveStream.close();
  renderRunSummary("Previous run results");
  const request = new XMLHttpRequest();
  videoUpload = request;
  request.open("POST", "/api/videos");
  request.setRequestHeader("Content-Type", "application/octet-stream");
  request.setRequestHeader("X-Video-Name", encodeURIComponent(file.name));
  document.getElementById("video-analyze").disabled = true;
  document.getElementById("video-input").disabled = true;
  document.getElementById("live-restart").disabled = true;
  document.getElementById("video-cancel").hidden = false;
  setLiveStatus("Previous run stopped. Preparing the selected video.");
  showVideoSelection(`Uploading ${file.name}… 0%`);
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) showVideoSelection(event.loaded === event.total ? "Checking the video before analysis…" : `Uploading ${file.name}… ${Math.round(event.loaded / event.total * 100)}%`);
  };
  request.onload = () => {
    try {
      const result = JSON.parse(request.responseText);
      if (request.status !== 201) throw new Error(result.error || "The video could not be uploaded.");
      liveVideoId = result.video_id;
      liveSourceName = result.source;
      livePhone = false;
      document.getElementById("live-restart").textContent = "Restart run";
      document.getElementById("live-preview-empty").textContent = "The first frame appears after the decoder and optional model are ready.";
      document.getElementById("live-title").textContent = `Live controller trace — ${result.source}`;
      document.getElementById("scenario-name").textContent = result.source;
      document.getElementById("scenario-meta").textContent = liveConfig?.semantic_model ? "Selected video + ViT classifier" : "Selected video — motion and memory analysis";
      document.getElementById("simulation-section").hidden = true;
      document.getElementById("simulation-timeline").hidden = true;
      showVideoSelection(`Analyzing ${result.source} — ${result.codec}, ${result.width} × ${result.height} pixels.`);
      startLiveStream();
    } catch (error) {
      showVideoSelection(error.message, true);
      setLiveStatus("Video selection failed. Choose another video or restart the previous run.", true);
    }
  };
  request.onerror = () => showVideoSelection("Upload failed. Check that the local dashboard server is running and try again.", true);
  request.onabort = () => { showVideoSelection("Upload canceled. Select Analyze video to try again."); setLiveStatus("Upload canceled. Restart the previous run or choose another video."); };
  request.onloadend = () => {
    videoUpload = null;
    document.getElementById("video-input").disabled = false;
    document.getElementById("video-analyze").disabled = false;
    document.getElementById("live-restart").disabled = false;
    document.getElementById("video-cancel").hidden = true;
  };
  request.send(file);
});
document.getElementById("video-cancel").addEventListener("click", () => videoUpload?.abort());

function setLiveStatus(message, isError = false) {
  const status = document.getElementById("live-status");
  status.textContent = message;
  status.classList.toggle("is-error", isError);
}

function renderLiveFrame(record) {
  if (record.type !== "frame_result") return;
  liveFrameCount += 1;
  if (record.metadata_fallback) liveFallbackCount += 1;
  // Keep a bounded, image-free trace even for unbounded network streams.
  const { preview_jpeg, ...trace } = record;
  liveRecords.push(trace);
  if (liveRecords.length > 10000) liveRecords.shift();
  document.getElementById("live-download").disabled = false;
  if (preview_jpeg) {
    const image = document.getElementById("live-preview");
    image.src = `data:image/jpeg;base64,${preview_jpeg}`;
    image.hidden = false;
    document.getElementById("live-preview-empty").hidden = true;
  }
  document.getElementById("live-source-time").textContent = `Frame ${record.generation} · ${metricValue(record.source_seconds, " s")}`;
  const memory = record.memory || {};
  document.getElementById("live-pages").replaceChildren(...(record.pages || []).map((page) => {
    const tile = document.createElement("div");
    const tier = ["T0", "T1", "T2", "T3"].includes(page.tier) ? page.tier : "unknown";
    tile.className = `page-tile tier-${tier.toLowerCase()}`;
    tile.tabIndex = 0;
    tile.textContent = `${page.page_id} · ${page.tier ?? "—"}`;
    tile.title = `Page ${page.page_id}: ${page.tier ?? "unmapped"}, ${page.state}, activity ${metricValue(page.activity)}, ${page.ready ? "ready" : "not ready"}`;
    tile.setAttribute("aria-label", tile.title);
    return tile;
  }));
  const rescues = record.semantic_feedback?.actions?.length ?? 0;
  document.getElementById("live-decision").textContent = record.status !== "READY"
    ? `Controller status: ${record.status}. Inspect the saved trace for apply statuses.`
    : record.metadata_fallback
      ? "Metadata fallback: missing or unreliable codec evidence keeps pages conservative."
      : `${memory.hot ?? 0} of ${memory.resident ?? 0} resident pages in T0. ${rescues} semantic correction(s). ${record.kv_reuse ? `${record.kv_reuse.pages_reused} K/V page(s) reused; ${record.kv_reuse.pages_recomputed} refreshed. ` : ""}${record.transition_predicted ? "Scene transition predicted." : "Motion and temporal history determine placement."}`;
  const storage = record.tensor_storage || {};
  const comparison = record.baseline_comparison || {};
  const values = [
    ["Frames processed", liveFrameCount, livePhone ? `${record.capture_dropped_frames ?? 0} older capture frames skipped to stay current` : "Decoded and committed"],
    ["Fallbacks", liveFallbackCount, "No reliable motion metadata"],
    ["T0 / resident pages", `${memory.hot ?? 0} / ${memory.resident ?? 0}`, "Committed controller mapping"],
    ["Processing rate", metricValue(1 / record.end_to_end_frame_seconds, " fps"), "Decode + processing; excludes preview and pacing"],
  ];
  document.getElementById("live-metrics").replaceChildren(...values.map(([label, value, detail]) => {
    const card = document.createElement("article"); card.className = "metric metric-cyan";
    const title = document.createElement("span"); title.className = "metric-label"; title.textContent = label;
    const number = document.createElement("strong"); number.className = "metric-value"; number.textContent = value;
    const note = document.createElement("span"); note.className = "metric-detail"; note.textContent = detail;
    card.append(title, number, note);
    return card;
  }));
  const row = document.createElement("tr");
  const cells = [record.generation, record.motion_vectors ?? "—", record.metadata_fallback ? "Yes" : "No",
    storage.payload_bytes === undefined ? "—" : metricValue(storage.payload_bytes, " B"),
    record.kv_reuse ? record.kv_reuse.pages_reused : "Off",
    record.model_inference_seconds === undefined ? "—" : metricValue(record.model_inference_seconds * 1000, " ms"),
    comparison.same_top1 === undefined ? "—" : comparison.same_top1 ? "Top-1 agrees" : "Top-1 differs"];
  cells.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
  const body = document.getElementById("live-frame-body");
  body.prepend(row);
  while (body.children.length > 40) body.lastElementChild.remove();
  const path = record.inference_path === "stored_kv" ? "stored-K/V" : record.inference_path === "codec_only" ? "codec/controller" : "standard model";
  setLiveStatus(`Processing ${path} path · frame ${record.generation}`);
}

function startLiveStream() {
  if (liveStream) liveStream.close();
  liveFrameCount = 0;
  liveFallbackCount = 0;
  liveRecords = [];
  liveStartedAt = performance.now();
  document.getElementById("video-results").hidden = true;
  document.getElementById("live-download").disabled = true;
  document.getElementById("live-preview").hidden = true;
  document.getElementById("live-preview").removeAttribute("src");
  document.getElementById("live-preview-empty").hidden = false;
  document.getElementById("live-source-time").textContent = "Waiting for input";
  document.getElementById("live-pages").replaceChildren();
  document.getElementById("live-decision").textContent = "Waiting for the first controller decision.";
  document.getElementById("live-frame-body").replaceChildren();
  document.getElementById("live-metrics").replaceChildren();
  setLiveStatus(livePhone ? "Connecting to Android camera over USB… Check the phone's authorization prompt." : "Connecting to local decoder…");
  const stream = new EventSource(liveVideoId ? `/api/stream?video=${encodeURIComponent(liveVideoId)}` : "/api/stream");
  liveStream = stream;
  stream.onmessage = (event) => {
    try { renderLiveFrame(JSON.parse(event.data)); }
    catch (error) { setLiveStatus(`Invalid frame record: ${error.message}`, true); stream.close(); }
  };
  stream.addEventListener("done", () => {
    renderRunSummary();
    setLiveStatus(`Run complete · ${liveFrameCount} frames in ${((performance.now() - liveStartedAt) / 1000).toFixed(1)} s. Save the trace or ${livePhone ? "reconnect the camera" : "restart to replay"}.`);
    stream.close();
  });
  stream.addEventListener("run_error", (event) => {
    renderRunSummary("Partial video results");
    setLiveStatus(`Run failed: ${JSON.parse(event.data).message}`, true);
    stream.close();
  });
  stream.onerror = () => {
    setLiveStatus("Connection lost. Restart the run to try again.", true);
    stream.close();
  };
}

document.getElementById("live-restart").addEventListener("click", startLiveStream);
document.getElementById("live-stop").addEventListener("click", () => {
  if (liveStream) liveStream.close();
  if (videoUpload) videoUpload.abort();
  renderRunSummary("Stopped run results");
  setLiveStatus(`Stopped after ${liveFrameCount} frames. ${livePhone ? "Connect camera to start a new live session." : "Restart begins at the start of the source."}`);
});
document.getElementById("live-download").addEventListener("click", () => {
  const blob = new Blob([liveRecords.map((record) => JSON.stringify(record)).join("\n") + "\n"], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mtr-kv-live.jsonl";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
if (window.location.protocol.startsWith("http")) {
  fetch("/api/config").then((response) => {
    if (!response.ok) throw new Error("dashboard server unavailable");
    return response.json();
  }).then((config) => {
    liveConfig = config;
    liveSourceName = config.source;
    document.getElementById("video-picker").hidden = !config.video_upload;
    document.getElementById("video-mode").textContent = config.semantic_model
      ? `Analysis includes the ViT classifier${config.kv_inference ? ", tiered K/V attention" : ""}${config.kv_reuse ? ", exact K/V reuse" : ""}${config.compare_baseline ? ", and comparison with standard ViT" : ""}.`
      : "Analysis includes motion vectors, memory tier decisions, fallback events, and processing times. The AI model is disabled for this server.";
    livePhone = config.source_kind === "android_usb_camera";
    document.getElementById("live-section").hidden = false;
    document.getElementById("simulation-section").hidden = true;
    document.getElementById("simulation-timeline").hidden = true;
    setLoadStatus("Live measurements are shown below. Load a simulator artifact to compare strategies separately.");
    document.getElementById("live-title").textContent = `Live controller trace · ${config.source}`;
    document.getElementById("scenario-name").textContent = config.source;
    document.getElementById("scenario-meta").textContent = config.semantic_model
      ? `Measured video + ViT classifier · ${config.kv_inference ? "K/V consumed in attention" : "K/V stored after attention"}`
      : "Measured codec/controller run · model disabled";
    if (livePhone) {
      document.getElementById("live-restart").textContent = "Connect camera";
      document.getElementById("live-preview-empty").textContent = "Connect your Android phone by USB, unlock it, and approve USB debugging. Then select Connect camera.";
      document.getElementById("scenario-meta").textContent = config.semantic_model ? "Live Android camera + ViT classifier" : "Live Android camera · codec/controller mode";
      setLiveStatus("Ready. Select Connect camera to start the phone's camera feed.");
    } else {
      startLiveStream();
    }
  }).catch(() => {});
}

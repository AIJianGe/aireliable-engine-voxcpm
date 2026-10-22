/* ============================================================
   VoxCPM TTS WebUI 逻辑（原生 JS，直连引擎 API）
   四模式联动：tts / design / clone / ultimate_clone
   设置区（模式/变体/加载） ↔ 参数区（模式专属/文本/高级） ↔ 结果区（SSE + 播放器）
   ============================================================ */
"use strict";

const API = "";  // 同源直连（引擎自身挂载 /ui）

// ---------- 状态 ----------
const state = {
  mode: "tts",
  speaker: null,   // { name, path, localUrl }（可控克隆参考音频，path 为引擎侧绝对路径）
  prompt: null,    // { name, path, localUrl }（极致克隆提示音频）
  variants: [],    // capabilities.variants
  generating: false,
  loading: false,
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

// SSE 阶段 → 中文（stage 定义见 progress_reporter.py ProgressStage）
const STAGE_TEXT = {
  "tts.start": "开始推理",
  "tts.model_ready": "模型就绪",
  "tts.text_processing": "文本处理",
  "tts.generating": "生成音频",
  "tts.saving": "保存产物",
  "completed": "完成",
};

// 模式提示（对应 Gradio ui.py 四 Tab）
const MODE_HINTS = {
  tts: "基础 TTS（tts）：输入文本直接合成语音（默认音色）",
  design: "声音设计（design）：用文字描述音色，如「一个大叔沙哑声音，温和又带些奸诈」",
  clone: "可控克隆（clone）：上传参考音频，克隆音色合成新文本",
  ultimate_clone: "极致克隆（ultimate_clone）：提示音频 + 精确转写，最大化还原音色细节",
};

const MODE_CN = {
  tts: "基础 TTS",
  design: "声音设计",
  clone: "可控克隆",
  ultimate_clone: "极致克隆",
};

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  bindModeSeg();
  bindSliders();
  bindUploads();
  bindButtons();
  refreshStatus();
  refreshCapabilities();
  setInterval(refreshStatus, 15000);  // 状态轮询（空闲时）
});

// ============================================================
// 设置区：模式切换（四模式联动显隐）
// ============================================================
function bindModeSeg() {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      $("mode-hint").textContent = MODE_HINTS[state.mode];
      // 模式专属区块联动显隐（对应 Gradio 四 Tab）
      $("design-field").style.display = state.mode === "design" ? "" : "none";
      $("clone-field").style.display = state.mode === "clone" ? "" : "none";
      $("ultimate-field").style.display = state.mode === "ultimate_clone" ? "" : "none";
      // 极致克隆时文本框标签语义变化
      $("text-label").innerHTML = state.mode === "ultimate_clone"
        ? '目标合成文本 <span class="req">（必填）</span>'
        : '合成文本 <span class="req">（必填）</span>';
      $("footer-mode").textContent = `模式：${MODE_CN[state.mode]}`;
    });
  });
  $("footer-mode").textContent = "模式：基础 TTS";
}

// ============================================================
// 参数区：滑块联动
// ============================================================
function bindSliders() {
  const pairs = [
    ["cfg-value", "cfg-value-val", (v) => parseFloat(v).toFixed(1)],
    ["timesteps", "timesteps-val", (v) => v],
  ];
  for (const [id, valId, fmt] of pairs) {
    $(id).addEventListener("input", (e) => { $(valId).textContent = fmt(e.target.value); });
  }
}

// ============================================================
// 参数区：音频上传（可控克隆参考 / 极致克隆提示）
// ============================================================
function bindUploads() {
  bindAudioDropzone("speaker-dropzone", "speaker-file-input", uploadSpeaker);
  bindAudioDropzone("prompt-dropzone", "prompt-file-input", uploadPrompt);
}

function bindAudioDropzone(dzId, fiId, handler) {
  const dz = $(dzId);
  const fi = $(fiId);
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    handler(e.dataTransfer.files[0]);
  });
  fi.addEventListener("change", () => handler(fi.files[0]));
}

async function uploadSpeaker(file) {
  if (!file) return;
  if (state.speaker) { showError("已有参考音频，请先移除后再上传"); return; }
  const localUrl = URL.createObjectURL(file);
  state.speaker = { name: file.name, path: "", localUrl };
  renderUploadChip("speaker", file.name, "上传中…");
  try {
    const data = await uploadFile(file);
    state.speaker.path = data.path;
    renderUploadChip("speaker", file.name, fmtSize(data.size ?? file.size));
  } catch (err) {
    removeUpload("speaker");
    showError(`参考音频上传失败：${err.message}`);
  }
}

async function uploadPrompt(file) {
  if (!file) return;
  if (state.prompt) { showError("已有提示音频，请先移除后再上传"); return; }
  const localUrl = URL.createObjectURL(file);
  state.prompt = { name: file.name, path: "", localUrl };
  renderUploadChip("prompt", file.name, "上传中…");
  try {
    const data = await uploadFile(file);
    state.prompt.path = data.path;
    renderUploadChip("prompt", file.name, fmtSize(data.size ?? file.size));
  } catch (err) {
    removeUpload("prompt");
    showError(`提示音频上传失败：${err.message}`);
  }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

function renderUploadChip(kind, name, sizeText) {
  const box = $(`${kind}-chip-box`);
  box.innerHTML = `
    <span class="audio-chip">
      <span class="fc-name" title="${name}">${kind === "speaker" ? "🎙" : "🎭"} ${name}</span>
      <span style="color:#1a7f37">${sizeText}</span>
      <button class="fc-rm" title="移除">×</button>
    </span>`;
  box.querySelector(".fc-rm").addEventListener("click", () => removeUpload(kind));
}

function removeUpload(kind) {
  const entry = kind === "speaker" ? state.speaker : state.prompt;
  if (entry?.localUrl?.startsWith("blob:")) URL.revokeObjectURL(entry.localUrl);
  if (kind === "speaker") state.speaker = null;
  else state.prompt = null;
  clearUploadChip(kind);
}

function clearUploadChip(kind) {
  $(`${kind}-chip-box`).innerHTML = "";
  $(`${kind}-file-input`).value = "";
}

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return Math.round(bytes / 1024) + " KB";
}

// ============================================================
// 设置区：变体 / 加载 / 卸载
// ============================================================
async function refreshCapabilities() {
  try {
    const r = await fetch(`${API}/capabilities`);
    const cap = await r.json();
    state.variants = cap.variants || [];
    renderVariantSelect();
  } catch {
    $("mode-hint").textContent = "capabilities 获取失败（引擎未启动？）";
  }
}

function renderVariantSelect() {
  const sel = $("variant-select");
  const cur = sel.value;
  sel.innerHTML = `<option value="">自动（默认变体）</option>`;
  for (const v of state.variants) {
    const opt = document.createElement("option");
    opt.value = v.name;
    const sr = v.sample_rate ? ` · ${Math.round(v.sample_rate / 1000)}kHz` : "";
    const tag = v.available ? "" : "（未下载）";
    opt.textContent = `${v.name}${sr}${tag}`;
    if (!v.available) opt.disabled = true;
    sel.appendChild(opt);
  }
  if (cur && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
}

function bindButtons() {
  $("btn-refresh-status").addEventListener("click", refreshStatus);
  $("btn-load").addEventListener("click", loadModel);
  $("btn-unload").addEventListener("click", unloadModel);
  $("btn-generate").addEventListener("click", generate);
}

// variant 可为 null（自动模式，引擎回退 default_variant → 第一个可用）
async function loadModel() {
  if (state.loading) return;
  state.loading = true;
  $("btn-load").disabled = true;
  $("btn-load").textContent = "加载中…";
  try {
    const variant = $("variant-select").value || null;
    const r = await fetch(`${API}/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型加载失败：${err.message}`);
  } finally {
    state.loading = false;
    $("btn-load").disabled = false;
    $("btn-load").textContent = "加载模型";
    refreshStatus();
  }
}

async function unloadModel() {
  try {
    const r = await fetch(`${API}/unload`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型卸载失败：${err.message}`);
  } finally {
    refreshStatus();
  }
}

// ============================================================
// 状态轮询
// ============================================================
async function refreshStatus() {
  if (state.generating) return;
  try {
    const r = await fetch(`${API}/status`);
    const st = await r.json();
    const badge = $("state-badge");
    badge.className = "badge " + (st.state || "");
    badge.textContent = {
      idle: "空闲", loading: "加载中", ready: "就绪",
      inferring: "推理中", error: "异常",
    }[st.state] || st.state;
    // VoxCPM 状态字段（loaded_variant / vram_used，extra 扩展）
    const loaded = normalizeVariantName(st.loaded_variant || st.extra?.loaded_variant || "");
    const vram = st.extra?.vram_used || st.vram_used;
    const dev = st.extra?.device_name || st.extra?.device || "—";
    $("state-detail").textContent =
      `变体 ${loaded || "未加载"} · 显存 ${vram || "—"} · ${dev}`;
    return st;
  } catch {
    $("state-badge").className = "badge";
    $("state-badge").textContent = "未连接";
    $("state-detail").textContent = "引擎不可达（127.0.0.1:7003）";
    return null;
  }
}

// variant 名归一化（对齐 ui.py：路径取最后一段 + 空格截断）
function normalizeVariantName(raw) {
  return (raw || "").replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop().split(" ")[0];
}

// ============================================================
// 结果区：生成（auto-load + SSE 流）
// ============================================================
async function generate() {
  if (state.generating) return;

  // 校验（对应 ui.py generate_tts 的模式校验）
  const text = $("text-input").value.trim();
  if (!text) { showError("请输入合成文本"); return; }
  if (state.mode === "design" && !$("design-desc").value.trim()) {
    showError("声音设计模式需要输入音色描述"); return;
  }
  if (state.mode === "clone") {
    if (!state.speaker?.path) { showError("可控克隆模式需要上传参考音频"); return; }
  }
  if (state.mode === "ultimate_clone") {
    if (!state.prompt?.path) { showError("极致克隆模式需要上传提示音频"); return; }
    if (!$("prompt-text").value.trim()) { showError("极致克隆模式需要输入转写文本"); return; }
  }

  // ----- 1. auto-load（对齐 ui.py：未加载自动 /load，variant 不匹配自动切换） -----
  const variant = $("variant-select").value || null;
  const st = await refreshStatus();
  if (st) {
    const loaded = normalizeVariantName(st.loaded_variant || st.extra?.loaded_variant || "");
    if (st.state !== "ready" || !loaded) {
      setProgress("正在加载模型…", 0);
      try {
        await postLoad(variant);
      } catch (err) {
        showError(`自动加载失败：${err.message}`);
        return;
      }
    } else if (variant && loaded !== variant) {
      setProgress(`正在切换模型 ${loaded} → ${variant}…`, 0);
      try {
        await fetch(`${API}/unload`, { method: "POST" });
        await postLoad(variant);
      } catch (err) {
        showError(`模型切换失败：${err.message}`);
        return;
      }
    }
  }

  // ----- 2. 构造请求体（对应 ui.py _build 逻辑） -----
  const inputs = {
    text,
    mode: state.mode,
    cfg_value: parseFloat($("cfg-value").value),
    inference_timesteps: parseInt($("timesteps").value, 10),
    seed: parseInt($("seed").value, 10) || -1,
    output_format: $("format-select").value,
  };
  const audios = [];
  if (state.mode === "design") {
    inputs.voice_description = $("design-desc").value.trim();
  } else if (state.mode === "clone") {
    audios.push({ path: state.speaker.path, role: "speaker" });
  } else if (state.mode === "ultimate_clone") {
    audios.push({ path: state.prompt.path, role: "prompt" });
    inputs.prompt_text = $("prompt-text").value.trim();
  }

  // 不携带 task_id：直连链路契约（引擎以 request.task_id 是否非空区分
  // gateway 转发/直连）。引擎会自动生成 task_id 上报任务中心全生命周期。
  const body = { inputs, audios };
  if (variant) body.variant = variant;

  // ----- 3. SSE 流式推理 -----
  state.generating = true;
  $("btn-generate").disabled = true;
  $("btn-generate").textContent = "合成中…";
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "连接引擎…";
  $("error-box").style.display = "none";
  $("state-badge").className = "badge inferring";
  $("state-badge").textContent = "推理中";

  try {
    const resp = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${resp.status}`);
    }

    // 解析 SSE 流（fetch ReadableStream 手动读 data: 行）
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResult = null;
    let finalTiming = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();  // 末行可能不完整，留在缓冲
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        let frame;
        try { frame = JSON.parse(line.slice(5)); } catch { continue; }
        const r = handleFrame(frame);
        if (r) { finalResult = r.result; finalTiming = r.timing; }
      }
    }

    if (finalResult) {
      showResult(finalResult, finalTiming);
    } else {
      throw new Error("SSE 流结束但未收到结果帧");
    }
  } catch (err) {
    showError(`合成失败：${err.message}`);
  } finally {
    state.generating = false;
    $("btn-generate").disabled = false;
    $("btn-generate").textContent = "生成语音";
    $("progress-wrap").style.display = "none";
    refreshStatus();
  }
}

async function postLoad(variant) {
  const r = await fetch(`${API}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ variant }),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
}

function setProgress(text, pct) {
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = `${pct}%`;
  $("progress-text").textContent = text;
}

// 处理一帧，返回最终结果（若有）
function handleFrame(f) {
  if (f.stage === "error") {
    throw new Error(f.error?.message || "引擎错误");
  }
  const pct = Math.round((f.progress || 0) * 100);
  $("progress-fill").style.width = `${pct}%`;
  const stageText = STAGE_TEXT[f.stage] || f.stage;
  const detail = f.message || "";
  $("progress-text").textContent = `${stageText} ${pct}%${detail ? " · " + detail : ""}`;
  if (f.stage === "completed" && f.result) {
    return { result: f.result, timing: f.timing };
  }
  return null;
}

function showResult(result, timing) {
  const view = $("result-view");
  const t = Date.now();  // 防缓存
  // result: {audio_url, audio_path, duration, sample_rate, mode}
  const audioUrl = result.audio_url + "?t=" + t;
  view.innerHTML = `
    <div class="audio-result">
      <span class="media-label">合成语音（${MODE_CN[result.mode] || result.mode} 模式）</span>
      <audio src="${audioUrl}" controls autoplay></audio>
    </div>`;

  $("result-meta").innerHTML = `
    <span class="meta-chip">模式 ${MODE_CN[result.mode] || result.mode}</span>
    <span class="meta-chip">时长 ${result.duration?.toFixed(1) ?? "—"}s</span>
    <span class="meta-chip">采样率 ${result.sample_rate ?? "—"}Hz</span>
    <span class="meta-chip">总耗时 ${timing?.total_seconds?.toFixed(1) ?? "—"}s</span>
    <span class="meta-chip">seed ${$("seed").value}</span>
    <span class="meta-chip"><a href="${audioUrl}" download>下载音频</a></span>`;
}

// ---------- 错误提示（不自动消失，手动关闭） ----------
function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}

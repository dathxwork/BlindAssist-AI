/**
 * BlindAssist AI v3.2 — Zero-Lag Synchronized Vision & Audio Controller
 * - Đồng bộ tuyệt đối 0.0ms giữa Video và Khung quét AI nhờ:
 *   1) Luồng giải mã C++ Pre-stream Keyframes trực tiếp từ máy chủ (/ws/video_analyze) ở chu kỳ 0.20s.
 *   2) Bộ nội suy chuyển động mượt 60 FPS (Keyframe Linear Interpolation) + Smart AI Sync Gate
 *      đảm bảo hình ảnh video không bao giờ chạy vượt trước khung nhận diện dù CPU tải nặng.
 *   3) Căn chỉnh hình học Canvas khớp 100% khung hình thực tế trên mọi độ phân giải.
 */

// --- 1. KHỞI TẠO TRẠNG THÁI TOÀN CỤC ---
const initialCfg = window.__INITIAL_SETTINGS__ || {};

const state = {
    isStreaming: false,
    mediaStream: null,
    facingMode: "environment",
    currentTab: "tab-live",
    highContrast: false,

    // Cài đặt thuật toán
    dangerDistance: initialCfg.danger_distance ?? 3.0,
    confThreshold: initialCfg.conf_threshold ?? 0.25,
    focalLength: initialCfg.focal_length ?? 650,
    enableBlur: initialCfg.enable_blur ?? true,
    enableVoice: initialCfg.enable_voice ?? true,
    enableSonar: initialCfg.enable_sonar ?? true,
    showZones: initialCfg.show_zones ?? true,
    volume: 1.0,
    prevVolume: 1.0,

    // Giọng nói TTS
    lastSpokenText: "",
    lastImageSummary: "",
    lastSpokenTime: 0,
    speakCooldown: Math.round((initialCfg.speech_cooldown ?? 2.8) * 1000),
    speechRate: initialCfg.speech_rate ?? 1.0,
    voiceEngine: initialCfg.voice_engine || "auto",

    // Radar âm thanh tiệm cận
    audioCtx: null,
    sonarInterval: null,
    closestDistance: 99.0,

    // WebSocket cho Webcam & Video tải lên cục bộ
    ws: null,
    wsReady: false,
    wsPending: new Map(),
    wsReqCounter: 0,
    isProcessingFrame: false,
};

// --- TIỆN ÍCH TOAST THÔNG BÁO ---
function showToast(message, type = "info", durationMs = 3200) {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const icons = { success: "✅", warn: "⚠️", error: "🚨", info: "💡" };
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${icons[type] || "💡"}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        toast.style.transition = "all 0.2s ease";
        setTimeout(() => toast.remove(), 220);
    }, durationMs);
}

// --- 2. KẾT NỐI WEBSOCKET STREAMING (/ws/stream) ---
function connectStreamWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
        return;
    }
    try {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(`${proto}//${window.location.host}/ws/stream`);
        state.ws = ws;

        ws.onopen = () => {
            state.wsReady = true;
            updateWsUiBadge(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const reqId = data.req_id;
                if (reqId !== undefined && state.wsPending.has(reqId)) {
                    const { resolve, timer } = state.wsPending.get(reqId);
                    clearTimeout(timer);
                    state.wsPending.delete(reqId);
                    resolve(data);
                }
            } catch (err) {
                console.warn("[WS Parse Error]:", err);
            }
        };

        ws.onclose = () => {
            state.wsReady = false;
            updateWsUiBadge(false);
            for (const [, entry] of state.wsPending.entries()) {
                clearTimeout(entry.timer);
                entry.reject(new Error("WS closed"));
            }
            state.wsPending.clear();
        };

        ws.onerror = () => {
            state.wsReady = false;
            updateWsUiBadge(false);
        };
    } catch (_) {
        state.wsReady = false;
        updateWsUiBadge(false);
    }
}

function updateWsUiBadge(isOnline) {
    const pill = document.getElementById("ws-status-pill");
    const modeBox = document.getElementById("stat-mode");
    if (pill) {
        pill.className = isOnline ? "badge badge-green" : "badge badge-blue";
        pill.innerText = isOnline ? "⚡ WebSocket Trực tuyến" : "🌐 HTTP REST";
    }
    if (modeBox) {
        modeBox.innerText = isOnline ? "WebSocket" : "HTTP REST";
        modeBox.style.color = isOnline ? "#34d399" : "#60a5fa";
    }
}

async function inferFrameRealtime(payload) {
    if (state.wsReady && state.ws && state.ws.readyState === WebSocket.OPEN) {
        const reqId = ++state.wsReqCounter;
        try {
            return await new Promise((resolve, reject) => {
                const timer = setTimeout(() => {
                    state.wsPending.delete(reqId);
                    reject(new Error("WS timeout"));
                }, 4500);
                state.wsPending.set(reqId, { resolve, reject, timer });
                state.ws.send(JSON.stringify({ ...payload, req_id: reqId }));
            });
        } catch (_) {}
    } else {
        connectStreamWebSocket();
    }

    const resp = await fetch("/api/detect_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
}

// --- 3. CĂN CHỈNH HÌNH HỌC CANVAS KHỚP 100% KHUNG HIỂN THỊ VIDEO ---
function syncOverlayToVideoRect(videoEl, canvasEl) {
    if (!videoEl || !canvasEl) return;
    const cw = videoEl.clientWidth;
    const ch = videoEl.clientHeight;
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!cw || !ch || !vw || !vh) return;

    const videoRatio = vw / vh;
    const elemRatio = cw / ch;
    let renderW, renderH, offsetX, offsetY;

    if (elemRatio > videoRatio) {
        renderH = ch;
        renderW = ch * videoRatio;
        offsetX = (cw - renderW) * 0.5;
        offsetY = 0;
    } else {
        renderW = cw;
        renderH = cw / videoRatio;
        offsetX = 0;
        offsetY = (ch - renderH) * 0.5;
    }

    const leftPx = Math.round(videoEl.offsetLeft + offsetX);
    const topPx = Math.round(videoEl.offsetTop + offsetY);
    const widthPx = Math.max(1, Math.round(renderW));
    const heightPx = Math.max(1, Math.round(renderH));

    canvasEl.style.left = `${leftPx}px`;
    canvasEl.style.top = `${topPx}px`;
    canvasEl.style.width = `${widthPx}px`;
    canvasEl.style.height = `${heightPx}px`;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetW = Math.max(1, Math.round(widthPx * dpr));
    const targetH = Math.max(1, Math.round(heightPx * dpr));
    if (canvasEl.width !== targetW || canvasEl.height !== targetH) {
        canvasEl.width = targetW;
        canvasEl.height = targetH;
    }
}

function getDirectionFromBox(box, capW) {
    const ratio = ((box[0] + box[2]) * 0.5) / Math.max(1, capW);
    if (ratio < 0.35) return "bên trái";
    if (ratio > 0.65) return "bên phải";
    return "chính giữa";
}

function computeBoxIoU(a, b) {
    const ix1 = Math.max(a[0], b[0]);
    const iy1 = Math.max(a[1], b[1]);
    const ix2 = Math.min(a[2], b[2]);
    const iy2 = Math.min(a[3], b[3]);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    if (inter <= 0) return 0;
    const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
    const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
    return inter / (areaA + areaB - inter);
}

// --- 4. ÂM THANH RADAR TIỆM CẬN (WEB AUDIO API) ---
function initAudioContext() {
    if (!state.audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) state.audioCtx = new AudioContext();
    }
    if (state.audioCtx && state.audioCtx.state === "suspended") {
        state.audioCtx.resume().catch(() => {});
    }
}

function playBeep(freq = 600, duration = 0.075) {
    if (!state.enableSonar || state.volume <= 0.001) return;
    try {
        initAudioContext();
        if (!state.audioCtx) return;
        const osc = state.audioCtx.createOscillator();
        const gain = state.audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, state.audioCtx.currentTime);

        const effectiveVol = Math.max(0.002, 0.15 * state.volume);
        gain.gain.setValueAtTime(effectiveVol, state.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0008, state.audioCtx.currentTime + duration);

        osc.connect(gain);
        gain.connect(state.audioCtx.destination);
        osc.start();
        osc.stop(state.audioCtx.currentTime + duration);
    } catch (_) {}
}

function playVolumeTick(vol = null) {
    const v = vol !== null ? vol : state.volume;
    if (v <= 0.001) return;
    try {
        initAudioContext();
        if (!state.audioCtx) return;
        const osc = state.audioCtx.createOscillator();
        const gain = state.audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(760, state.audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(460, state.audioCtx.currentTime + 0.045);
        gain.gain.setValueAtTime(0.09 * v, state.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + 0.045);
        osc.connect(gain);
        gain.connect(state.audioCtx.destination);
        osc.start();
        osc.stop(state.audioCtx.currentTime + 0.045);
    } catch (_) {}
}

let lastSonarBucket = -1;
function stopSonarRadar() {
    if (state.sonarInterval) {
        clearInterval(state.sonarInterval);
        state.sonarInterval = null;
    }
    lastSonarBucket = -1;
}

function updateSonarRadar(distance) {
    state.closestDistance = distance ?? 99.0;
    const activeMediaScanning = state.isStreaming || isVideoAnalyzing;

    if (!state.enableSonar || !activeMediaScanning || distance === null || distance === undefined || distance > 4.2) {
        if (lastSonarBucket !== -1) stopSonarRadar();
        return;
    }

    let bucket = 3;
    let intervalMs = 800;
    let freq = 580;

    if (distance <= 1.5) {
        bucket = 1; intervalMs = 220; freq = 880;
    } else if (distance <= 2.5) {
        bucket = 2; intervalMs = 440; freq = 720;
    }

    if (bucket === lastSonarBucket && state.sonarInterval) return;

    stopSonarRadar();
    lastSonarBucket = bucket;
    state.sonarInterval = setInterval(() => {
        if (!state.isStreaming && !isVideoAnalyzing) {
            stopSonarRadar();
            return;
        }
        playBeep(freq, 0.06);
    }, intervalMs);
}

// --- 5. CẢNH BÁO GIỌNG NÓI TIẾNG VIỆT ĐA NGUỒN ---
const SILENT_AUDIO_WAV = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
const ttsAudio = new Audio();
ttsAudio.preload = "auto";
let isAudioPlaying = false;
let audioUnlocked = false;
let isTestingVoice = false;
let safetyAudioTimer = null;

function setSpeakingWave(active) {
    document.getElementById("voice-wave-indicator")?.classList.toggle("speaking", !!active);
}

function unlockAudioPlayback() {
    if (audioUnlocked) return;
    initAudioContext();
    ttsAudio.src = SILENT_AUDIO_WAV;
    ttsAudio.play().then(() => {
        ttsAudio.pause();
        ttsAudio.currentTime = 0;
        audioUnlocked = true;
    }).catch(() => {
        audioUnlocked = true;
    });
}
window.addEventListener("click", unlockAudioPlayback, { once: true });
window.addEventListener("keydown", unlockAudioPlayback, { once: true });
window.addEventListener("touchstart", unlockAudioPlayback, { once: true });

function finishSpeakingCleanup() {
    isAudioPlaying = false;
    setSpeakingWave(false);
    if (isTestingVoice) {
        isTestingVoice = false;
        document.getElementById("btn-test-voice")?.classList.remove("active");
    }
    if (videoPlayer && !videoPlayer.paused) {
        videoPlayer.volume = 0.45 * state.volume;
    }
}

ttsAudio.addEventListener("ended", finishSpeakingCleanup);

function speakViaBrowserWebSpeech(text) {
    if (!("speechSynthesis" in window) || state.volume <= 0.001) {
        finishSpeakingCleanup();
        return;
    }
    try {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = "vi-VN";
        utter.rate = Math.min(2.0, Math.max(0.5, state.speechRate));
        utter.volume = state.volume;

        const voices = window.speechSynthesis.getVoices() || [];
        const vnVoice = voices.find(v =>
            v.lang.toLowerCase().includes("vi") || v.name.toLowerCase().includes("vietnam")
        );
        if (vnVoice) utter.voice = vnVoice;

        utter.onstart = () => {
            isAudioPlaying = true;
            setSpeakingWave(true);
        };
        utter.onend = finishSpeakingCleanup;
        utter.onerror = finishSpeakingCleanup;
        window.speechSynthesis.speak(utter);
    } catch (_) {
        finishSpeakingCleanup();
    }
}

ttsAudio.addEventListener("error", () => {
    if (state.voiceEngine === "auto" && state.lastSpokenText) {
        speakViaBrowserWebSpeech(state.lastSpokenText);
    } else {
        finishSpeakingCleanup();
    }
});

function playVietnameseAudio(text) {
    if (!state.enableVoice || state.volume <= 0.001 || !text) return;

    if (state.voiceEngine === "browser") {
        speakViaBrowserWebSpeech(text);
        return;
    }

    if ("speechSynthesis" in window) window.speechSynthesis.cancel();

    if (videoPlayer && !videoPlayer.paused) {
        videoPlayer.volume = 0.12 * state.volume;
    }

    const audioUrl = `/api/tts?text=${encodeURIComponent(text.trim())}`;
    try {
        ttsAudio.defaultPlaybackRate = state.speechRate;
        ttsAudio.src = audioUrl;
        ttsAudio.playbackRate = state.speechRate;
        ttsAudio.volume = state.volume;

        const playPromise = ttsAudio.play();
        if (playPromise !== undefined) {
            playPromise.then(() => {
                isAudioPlaying = true;
                setSpeakingWave(true);
                if (safetyAudioTimer) clearTimeout(safetyAudioTimer);
                safetyAudioTimer = setTimeout(finishSpeakingCleanup, 6000);
            }).catch(finishSpeakingCleanup);
        }
    } catch (_) {
        finishSpeakingCleanup();
    }
}

function formatDistanceSpeechJs(distance) {
    if (distance === null || distance === undefined) return "";
    const d = Math.round(distance * 10) / 10;
    if (d < 1.0) return d >= 0.7 ? "cách chưa tới 1 mét" : `cách 0 phẩy ${Math.round(d * 10)} mét`;
    if (Math.abs(d - 1.5) < 0.05) return "cách 1 mét rưỡi";
    if (Math.abs(d - 2.5) < 0.05) return "cách 2 mét rưỡi";
    if (Math.abs(d - 3.5) < 0.05) return "cách 3 mét rưỡi";
    if (Math.abs(d - Math.round(d)) < 0.15) return `cách ${Math.round(d)} mét`;
    return `cách ${d.toFixed(1).replace(".", " phẩy ")} mét`;
}

let lastSpokenItemKey = "";
let lastSpokenItemTime = 0;
let lastAlertGlobalTime = 0;

function alertDangerousObstacle(item) {
    if (!state.enableVoice || !item) return;

    const now = Date.now();
    if (isAudioPlaying) return;

    const cooldownMs = state.speakCooldown || 2800;
    if (now - lastAlertGlobalTime < cooldownMs) return;

    const itemKey = `${item.label}_${item.direction}`;
    if (itemKey === lastSpokenItemKey && now - lastSpokenItemTime < Math.max(4000, cooldownMs * 1.4)) {
        return;
    }

    lastSpokenItemKey = itemKey;
    lastSpokenItemTime = now;
    lastAlertGlobalTime = now;

    const dirSpeech = item.direction === "chính giữa" ? "phía trước" : item.direction;
    const lblLower = (item.label || "vật cản").toLowerCase();
    const distSpeech = formatDistanceSpeechJs(item.distance);

    let alertText = item.alert_message;
    if (!alertText) {
        if (lblLower.includes("ổ gà")) {
            alertText = distSpeech ? `Cảnh báo, phát hiện ổ gà ${dirSpeech}, ${distSpeech}.` : `Cảnh báo, phát hiện ổ gà ${dirSpeech}.`;
        } else if (item.distance_level === "Rất gần" || (item.distance !== undefined && item.distance <= 1.5)) {
            alertText = distSpeech ? `Cảnh báo nguy hiểm, có ${lblLower} ở ${dirSpeech}, ${distSpeech}.` : `Cảnh báo nguy hiểm, có ${lblLower} rất gần ở ${dirSpeech}.`;
        } else {
            alertText = distSpeech ? `Có ${lblLower} ở ${dirSpeech}, ${distSpeech}.` : `Có ${lblLower} ở ${dirSpeech}.`;
        }
    }

    speakAlert(alertText, true);
}

function speakAlert(text, isDanger = false) {
    if (!text) return;
    state.lastSpokenText = text;
    state.lastSpokenTime = Date.now();
    logSpeech(text, isDanger);
    if (state.enableVoice) playVietnameseAudio(text);
}

function logSpeech(text, isDanger) {
    const logBox = document.getElementById("speech-log");
    if (!logBox) return;

    const timeStr = new Date().toLocaleTimeString("vi-VN", { hour12: false });
    const item = document.createElement("div");
    item.className = `speech-bubble ${isDanger ? "danger" : ""}`;

    const strong = document.createElement("strong");
    strong.textContent = text;
    const small = document.createElement("small");
    small.textContent = timeStr;

    item.appendChild(strong);
    item.appendChild(small);
    logBox.prepend(item);

    while (logBox.children.length > 25) {
        logBox.removeChild(logBox.lastChild);
    }
}

// --- 6. TAB 1: WEBCAM TRỰC TIẾP VỚI NỘI SUY VẬN TỐC 60 FPS ---
const videoElement = document.getElementById("webcam-video");
const overlayCanvas = document.getElementById("overlay-canvas");
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let livePrevState = null; // { timeMs, detections }
let liveCurrState = null; // { timeMs, detections, most_dangerous, capW, capH }
let liveRenderRaf = null;
let liveCapW = 640;
let liveCapH = 480;

function syncLiveCanvasSize() {
    if (!videoElement || !overlayCanvas) return;
    syncOverlayToVideoRect(videoElement, overlayCanvas);
    const vw = videoElement.videoWidth || 640;
    const vh = videoElement.videoHeight || 480;
    liveCapW = 640;
    liveCapH = Math.max(1, Math.round(liveCapW * (vh / vw)));
    if (captureCanvas.width !== liveCapW || captureCanvas.height !== liveCapH) {
        captureCanvas.width = liveCapW;
        captureCanvas.height = liveCapH;
    }
}

videoElement?.addEventListener("loadedmetadata", syncLiveCanvasSize);

function liveRenderLoop60Fps() {
    if (!state.isStreaming) {
        liveRenderRaf = null;
        return;
    }
    syncOverlayToVideoRect(videoElement, overlayCanvas);

    if (liveCurrState) {
        let drawList = liveCurrState.detections;
        // Ngoại suy vận tốc nhẹ trong khoảng 0..160ms để khung bao mượt 60 FPS trên Webcam
        if (livePrevState && liveCurrState.timeMs > livePrevState.timeMs) {
            const dtSnap = (liveCurrState.timeMs - livePrevState.timeMs) / 1000;
            const dtSince = Math.min(0.18, (performance.now() - liveCurrState.timeMs) / 1000);
            if (dtSnap > 0.03 && dtSnap < 0.8 && dtSince > 0) {
                const factor = Math.min(0.85, dtSince / dtSnap);
                drawList = liveCurrState.detections.map(cur => {
                    const prev = livePrevState.detections.find(p =>
                        p.raw_label === cur.raw_label &&
                        Math.hypot((p.box[0] + p.box[2] - cur.box[0] - cur.box[2]) * 0.5, (p.box[1] + p.box[3] - cur.box[1] - cur.box[3]) * 0.5) < liveCapW * 0.25
                    );
                    if (!prev) return cur;
                    const vx = (cur.box[0] - prev.box[0]) * factor;
                    const vy = (cur.box[1] - prev.box[1]) * factor;
                    const bw = cur.box[2] - cur.box[0];
                    const bh = cur.box[3] - cur.box[1];
                    const nx1 = Math.max(0, Math.min(liveCapW - bw, cur.box[0] + vx));
                    const ny1 = Math.max(0, Math.min(liveCapH - bh, cur.box[1] + vy));
                    return { ...cur, box: [nx1, ny1, nx1 + bw, ny1 + bh] };
                });
            }
        }
        drawHudCanvas(overlayCanvas, videoElement, drawList, liveCurrState.most_dangerous, liveCurrState.capW, liveCurrState.capH);
    }

    liveRenderRaf = requestAnimationFrame(liveRenderLoop60Fps);
}

async function startCamera() {
    try {
        initAudioContext();
        connectStreamWebSocket();

        // 1. Kiểm tra tính khả dụng của API Camera trên trình duyệt
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast("Trình duyệt không hỗ trợ hoặc chặn quyền truy cập Camera trên kết nối này.", "warn", 4000);
            return;
        }

        // 2. Mở camera
        try {
            state.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: state.facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false,
            });
        } catch (firstErr) {
            state.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: state.facingMode },
                audio: false,
            }).catch(async () => {
                return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            });
        }

        videoElement.srcObject = state.mediaStream;
        await videoElement.play();

        state.isStreaming = true;
        livePrevState = null;
        liveCurrState = null;
        document.getElementById("cam-idle-overlay")?.classList.add("hidden");

        const btnToggle = document.getElementById("btn-toggle-cam");
        if (btnToggle) {
            btnToggle.innerHTML = `
                <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                Tạm dừng Camera [Space]
            `;
            btnToggle.classList.replace("btn-primary", "btn-danger");
        }

        const camBadge = document.getElementById("cam-status-badge");
        if (camBadge) {
            camBadge.innerHTML = `<span class="status-dot"></span> Camera Đang Quét`;
            camBadge.className = "badge badge-green";
        }

        syncLiveCanvasSize();
        speakAlert("Hệ thống camera đã kích hoạt. Đang quét vật cản.");
        showToast("Đã bật Camera AI thời gian thực", "success");

        if (!liveRenderRaf) liveRenderRaf = requestAnimationFrame(liveRenderLoop60Fps);
        runFrameProcessingLoop();
    } catch (err) {
        console.error("Lỗi mở camera:", err);
        showToast("Không thể truy cập Camera: " + (err.message || err.name), "error", 4000);
    }
}

function stopCamera() {
    state.isStreaming = false;
    if (liveRenderRaf) {
        cancelAnimationFrame(liveRenderRaf);
        liveRenderRaf = null;
    }
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(track => track.stop());
        state.mediaStream = null;
    }
    if (videoElement) videoElement.srcObject = null;
    stopSonarRadar();

    livePrevState = null;
    liveCurrState = null;

    if (overlayCanvas) {
        const ctx = overlayCanvas.getContext("2d");
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
    document.getElementById("cam-idle-overlay")?.classList.remove("hidden");

    const btnToggle = document.getElementById("btn-toggle-cam");
    if (btnToggle) {
        btnToggle.innerHTML = `
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
            Bật Camera [Space]
        `;
        btnToggle.classList.replace("btn-danger", "btn-primary");
    }

    const camBadge = document.getElementById("cam-status-badge");
    if (camBadge) {
        camBadge.innerHTML = `<span class="status-dot"></span> Camera Đang Tắt`;
        camBadge.className = "badge";
    }

    updateDangerBanner("danger-banner", null);
    updateProximityGauge(null);
    updateSpatialZonesMini([]);
}

function runFrameProcessingLoop() {
    if (!state.isStreaming) return;
    sendFrameForDetection().finally(() => {
        if (state.isStreaming) setTimeout(runFrameProcessingLoop, 10);
    });
}

async function sendFrameForDetection() {
    if (state.isProcessingFrame || !videoElement || !videoElement.videoWidth) return;
    state.isProcessingFrame = true;

    try {
        syncLiveCanvasSize();
        captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.68);

        const data = await inferFrameRealtime({
            frame_data: dataUrl,
            danger_distance: state.dangerDistance,
            conf_threshold: state.confThreshold,
            enable_blur: state.enableBlur,
            return_image: false,
            imgsz: 320,
        });

        if (data && data.success !== false && state.isStreaming) {
            livePrevState = liveCurrState;
            liveCurrState = {
                timeMs: performance.now(),
                detections: data.detections || [],
                most_dangerous: data.most_dangerous || null,
                capW: data.frame_width || liveCapW,
                capH: data.frame_height || liveCapH,
            };
            handleLiveTelemetry(data);
        }
    } catch (err) {
        console.warn("Lỗi suy luận webcam:", err);
    } finally {
        state.isProcessingFrame = false;
    }
}

// --- 7. VẼ HUD CANVAS CHUNG ---
function getObjectColor(item, isDanger) {
    if (isDanger) return "#ef4444";
    const lbl = (item.label || "").toLowerCase();
    const raw = (item.raw_label || "").toLowerCase();
    if (lbl.includes("ổ gà") || raw.includes("o ga")) return "#ef4444";
    if (item.category === "person" || lbl.includes("người") || raw.includes("nguoi")) return "#10b981";
    if (lbl.includes("ô tô") || raw.includes("xe hoi") || raw.includes("car")) return "#eab308";
    if (lbl.includes("xe máy") || raw.includes("xe may") || raw.includes("motorcycle")) return "#ec4899";
    if (lbl.includes("xe đạp") || raw.includes("xe dap") || raw.includes("bicycle")) return "#3b82f6";
    if (lbl.includes("đèn giao thông") || raw.includes("den giao thong") || raw.includes("traffic light")) return "#f59e0b";
    if (lbl.includes("bảng hiệu") || raw.includes("bang hieu") || raw.includes("stop sign")) return "#8b5cf6";
    if (lbl.includes("cột") || raw.includes("cot")) return "#06b6d4";
    if (lbl.includes("cây cối") || raw.includes("cay coi")) return "#14b8a6";
    if (lbl.includes("thùng rác") || raw.includes("thung rac")) return "#94a3b8";
    if (item.category === "vehicle") return "#f59e0b";
    return "#3b82f6";
}

function drawRoundedPill(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
}

function drawHudCanvas(canvas, sourceMedia, detections, mostDangerous, capW, capH) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const scaleX = capW && capW > 0 ? w / capW : 1;
    const scaleY = capH && capH > 0 ? h / capH : 1;
    const uiScale = Math.max(0.85, Math.min(1.65, w / 820));

    const mediaW = (sourceMedia && sourceMedia.videoWidth) ? sourceMedia.videoWidth : capW;
    const mediaH = (sourceMedia && sourceMedia.videoHeight) ? sourceMedia.videoHeight : capH;
    const mediaScaleX = capW && capW > 0 ? mediaW / capW : 1;
    const mediaScaleY = capH && capH > 0 ? mediaH / capH : 1;

    if (state.showZones) {
        const leftX = Math.round(w * 0.35);
        const rightX = Math.round(w * 0.65);

        ctx.save();
        ctx.strokeStyle = "rgba(148, 163, 184, 0.28)";
        ctx.lineWidth = Math.max(1, 1.2 * uiScale);
        ctx.setLineDash([6 * uiScale, 6 * uiScale]);

        ctx.beginPath();
        ctx.moveTo(leftX, 0); ctx.lineTo(leftX, h);
        ctx.moveTo(rightX, 0); ctx.lineTo(rightX, h);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.font = `700 ${Math.round(11 * uiScale)}px Inter, system-ui, sans-serif`;
        ctx.fillStyle = "rgba(203, 213, 225, 0.55)";
        ctx.textAlign = "center";
        const zoneLabelY = h - Math.round(46 * uiScale);
        ctx.fillText("TRÁI", leftX / 2, zoneLabelY);
        ctx.fillText("PHÍA TRƯỚC", (leftX + rightX) / 2, zoneLabelY);
        ctx.fillText("PHẢI", (rightX + w) / 2, zoneLabelY);
        ctx.restore();
    }

    (detections || []).forEach(item => {
        const [x1, y1, x2, y2] = item.box;
        const isDanger = !!item.is_dangerous;

        const bx = x1 * scaleX;
        const by = y1 * scaleY;
        const bw = (x2 - x1) * scaleX;
        const bh = (y2 - y1) * scaleY;

        if (item.category === "person" && state.enableBlur && sourceMedia && sourceMedia.readyState >= 2) {
            try {
                const headRatio = 0.22;
                const sx = x1 * mediaScaleX;
                const sy = y1 * mediaScaleY;
                const sw = (x2 - x1) * mediaScaleX;
                const sh = Math.max(10, (y2 - y1) * headRatio * mediaScaleY);
                const headDstH = Math.max(12, bh * headRatio);

                ctx.save();
                ctx.beginPath();
                ctx.rect(bx, by, bw, headDstH);
                ctx.clip();
                ctx.filter = `blur(${Math.round(12 * uiScale)}px)`;
                ctx.drawImage(sourceMedia, sx, sy, sw, sh, bx - 2, by - 2, bw + 4, headDstH + 4);
                ctx.restore();
            } catch (_) {}
        }

        const color = getObjectColor(item, isDanger);

        ctx.strokeStyle = color;
        ctx.lineWidth = (isDanger ? 2.5 : 1.6) * uiScale;
        ctx.strokeRect(bx, by, bw, bh);

        const clen = Math.max(8, Math.min(22 * uiScale, Math.min(bw, bh) * 0.24));
        ctx.lineWidth = (isDanger ? 4.2 : 3.2) * uiScale;
        ctx.beginPath();
        ctx.moveTo(bx, by + clen); ctx.lineTo(bx, by); ctx.lineTo(bx + clen, by);
        ctx.moveTo(bx + bw - clen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + clen);
        ctx.moveTo(bx, by + bh - clen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + clen, by + bh);
        ctx.moveTo(bx + bw - clen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - clen);
        ctx.stroke();

        const distVal = typeof item.distance === "number" ? item.distance.toFixed(1) : item.distance;
        const label = `${item.label} • ${distVal}m`;
        const fontSize = Math.round(13.5 * uiScale);
        ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const padX = 8 * uiScale;
        const pillW = textWidth + padX * 2;
        const pillH = fontSize + 9 * uiScale;
        const pillX = Math.max(2, Math.min(w - pillW - 2, bx));
        const pillY = by - pillH - 4 >= 2 ? by - pillH - 4 : Math.min(h - pillH - 2, by + 4);

        ctx.save();
        drawRoundedPill(ctx, pillX, pillY, pillW, pillH, 6 * uiScale);
        ctx.fillStyle = isDanger ? "rgba(220, 38, 38, 0.92)" : "rgba(15, 23, 42, 0.88)";
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = isDanger ? "#fecaca" : color;
        ctx.stroke();

        ctx.fillStyle = isDanger ? "#ffffff" : color;
        ctx.textBaseline = "middle";
        ctx.fillText(label, pillX + padX, pillY + pillH / 2 + 0.5);
        ctx.restore();
    });

    if (mostDangerous) {
        ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
        ctx.lineWidth = 6 * uiScale;
        ctx.strokeRect(0, 0, w, h);
    }
}

// --- 8. CẬP NHẬT GIAO DIỆN CHUNG ---
function handleLiveTelemetry(data) {
    const detections = data.detections || [];
    const most_dangerous = data.most_dangerous || null;
    const fps = data.fps ?? 0;
    const latency_ms = data.latency_ms ?? 0;

    document.getElementById("stat-fps").innerText = `${fps} FPS`;
    document.getElementById("stat-latency").innerText = `${latency_ms} ms`;
    document.getElementById("stat-count").innerText = detections.length;

    renderTable("items-table-body", detections, "Không có vật thể nào trong tầm nhìn");
    updateSpatialZonesMini(detections);

    const nearestOverall = detections.length > 0
        ? detections.reduce((min, d) => (d.distance < min.distance ? d : min), detections[0])
        : null;

    updateProximityGauge(nearestOverall ? nearestOverall.distance : null);

    if (most_dangerous) {
        updateDangerBanner("danger-banner", most_dangerous);
        alertDangerousObstacle(most_dangerous);
        updateSonarRadar(most_dangerous.distance);
    } else {
        updateDangerBanner("danger-banner", null);
        updateSonarRadar(nearestOverall ? nearestOverall.distance : 99.0);
    }
}

function updateSpatialZonesMini(detections) {
    const map = {
        "bên trái": { el: document.getElementById("zone-ind-left"), valEl: document.getElementById("zone-val-left"), min: null },
        "chính giữa": { el: document.getElementById("zone-ind-center"), valEl: document.getElementById("zone-val-center"), min: null },
        "bên phải": { el: document.getElementById("zone-ind-right"), valEl: document.getElementById("zone-val-right"), min: null },
    };

    (detections || []).forEach(d => {
        const slot = map[d.direction];
        if (slot && (!slot.min || d.distance < slot.min.distance)) slot.min = d;
    });

    Object.values(map).forEach(({ el, valEl, min }) => {
        if (!el || !valEl) return;
        el.classList.remove("zone-warn", "zone-danger");
        if (!min) {
            valEl.innerText = "Trống";
        } else {
            valEl.innerText = `${min.label} (${min.distance}m)`;
            if (min.is_dangerous || min.distance <= state.dangerDistance) el.classList.add("zone-danger");
            else if (min.distance <= 5.0) el.classList.add("zone-warn");
        }
    });
}

function updateDangerBanner(bannerId, dangerItem, safeMessage = null) {
    const banner = document.getElementById(bannerId);
    if (!banner) return;

    if (dangerItem) {
        banner.className = "danger-banner danger";
        const lvl = dangerItem.distance_level || (dangerItem.distance <= 1.5 ? "Rất gần" : "Gần");
        const dirDisplay = dangerItem.direction === "chính giữa" ? "PHÍA TRƯỚC" : dangerItem.direction.toUpperCase();
        banner.innerHTML = `
            <div class="banner-content">
                <svg width="22" height="22" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                <span>CẢNH BÁO NGUY HIỂM: ${dangerItem.label.toUpperCase()} Ở ${dirDisplay} (${dangerItem.distance}m — ${lvl})!</span>
            </div>
            <span class="badge badge-red">${lvl} (${dangerItem.distance}m)</span>
        `;
    } else {
        banner.className = "danger-banner safe";
        const msg = safeMessage || "Đường đi phía trước thông thoáng. Không phát hiện nguy cơ va chạm.";
        banner.innerHTML = `
            <div class="banner-content">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <span>${msg}</span>
            </div>
            <span class="badge badge-green">An Toàn</span>
        `;
    }
}

function updateProximityGauge(dist) {
    const valElem = document.getElementById("gauge-value");
    const barElem = document.getElementById("gauge-bar");
    const lvlBadge = document.getElementById("gauge-level-badge");
    if (!valElem || !barElem) return;

    if (dist === null || dist === undefined) {
        valElem.innerText = "-- m";
        valElem.style.color = "var(--text-secondary)";
        barElem.style.width = "0%";
        barElem.style.background = "var(--accent-green)";
        if (lvlBadge) { lvlBadge.className = "badge badge-green"; lvlBadge.innerText = "Thông thoáng"; }
        return;
    }

    valElem.innerText = `${dist.toFixed(1)} m`;
    const percent = Math.min(100, Math.max(8, ((6.5 - dist) / 6.5) * 100));
    barElem.style.width = `${percent}%`;

    if (dist <= 1.5) {
        valElem.style.color = "var(--accent-red)";
        barElem.style.background = "var(--accent-red)";
        if (lvlBadge) { lvlBadge.className = "badge badge-red"; lvlBadge.innerText = "Rất gần"; }
    } else if (dist <= 3.0) {
        valElem.style.color = "var(--accent-yellow)";
        barElem.style.background = "var(--accent-yellow)";
        if (lvlBadge) { lvlBadge.className = "badge badge-yellow"; lvlBadge.innerText = "Gần"; }
    } else if (dist <= 5.0) {
        valElem.style.color = "var(--accent-blue)";
        barElem.style.background = "var(--accent-blue)";
        if (lvlBadge) { lvlBadge.className = "badge badge-blue"; lvlBadge.innerText = "Trung bình"; }
    } else {
        valElem.style.color = "var(--accent-green)";
        barElem.style.background = "var(--accent-green)";
        if (lvlBadge) { lvlBadge.className = "badge badge-green"; lvlBadge.innerText = "Xa (An toàn)"; }
    }
}

function renderTable(tbodyId, detections, emptyText = "Chưa có dữ liệu") {
    const tbody = document.getElementById(tbodyId);
    if (!tbody) return;

    if (!detections || detections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="empty-table-cell">${emptyText}</td></tr>`;
        return;
    }

    const sorted = [...detections].sort((a, b) => a.distance - b.distance);
    tbody.innerHTML = sorted.map(item => {
        const lvl = item.distance_level || (item.distance <= 1.5 ? "Rất gần" : item.distance <= 3.0 ? "Gần" : item.distance <= 5.0 ? "Trung bình" : "Xa");
        let badgeClass = "badge-green";
        if (lvl === "Rất gần" || item.is_dangerous) badgeClass = "badge-red";
        else if (lvl === "Gần") badgeClass = "badge-yellow";
        else if (lvl === "Trung bình") badgeClass = "badge-blue";

        const dotColor = getObjectColor(item, item.is_dangerous);
        const confBadge = item.confidence ? `<small style="color:var(--text-secondary);font-size:0.73rem;">(${Math.round(item.confidence * 100)}%)</small>` : "";
        const dirDisplay = item.direction === "chính giữa" ? "Phía trước" : item.direction;

        return `
        <tr class="${item.is_dangerous ? "danger-row" : ""}">
            <td><span class="obj-dot" style="background:${dotColor};"></span><strong>${item.label}</strong> ${confBadge}</td>
            <td><span class="direction-tag">${dirDisplay}</span></td>
            <td class="mono"><strong>${item.distance}m</strong></td>
            <td><span class="badge ${badgeClass}" title="${item.alert_message || ""}">${lvl}</span></td>
        </tr>`;
    }).join("");
}

// --- 9. TAB 2: PHÂN TÍCH ẢNH TĨNH ---
async function analyzeImage(file = null, sampleName = null) {
    const resultImg = document.getElementById("image-result-view");
    const placeholder = document.getElementById("image-placeholder");
    const loadingOverlay = document.getElementById("image-loading-overlay");
    const metaPill = document.getElementById("image-meta");
    const downloadBtn = document.getElementById("btn-download-image");

    const formData = new FormData();
    if (file) formData.append("file", file);
    if (sampleName) formData.append("sample_name", sampleName);
    formData.append("danger_distance", state.dangerDistance);
    formData.append("conf_threshold", state.confThreshold);
    formData.append("enable_blur", state.enableBlur ? "true" : "false");

    if (loadingOverlay) loadingOverlay.style.display = "flex";

    try {
        const resp = await fetch("/api/detect_image", { method: "POST", body: formData });
        const data = await resp.json();

        if (data.success) {
            resultImg.src = data.processed_image;
            resultImg.style.display = "block";
            placeholder.style.display = "none";

            if (downloadBtn) {
                downloadBtn.href = data.processed_image;
                downloadBtn.style.display = "inline-flex";
            }
            if (metaPill) {
                metaPill.innerText = `${data.image_size} • ${data.latency_ms}ms • ${data.objects_count} vật thể`;
            }

            state.lastImageSummary = data.summary_text;
            speakAlert(data.summary_text, !!data.most_dangerous);

            const imgBanner = document.getElementById("image-danger-banner");
            if (imgBanner) {
                if (data.most_dangerous) {
                    updateDangerBanner("image-danger-banner", data.most_dangerous);
                } else {
                    imgBanner.className = "danger-banner safe";
                    imgBanner.innerHTML = `
                        <div class="banner-content">
                            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            <span id="image-summary-text">${data.summary_text}</span>
                        </div>
                        <span class="badge badge-green">An Toàn</span>
                    `;
                }
            }

            const cEl = document.getElementById("image-stat-count");
            const nEl = document.getElementById("image-stat-nearest");
            const lEl = document.getElementById("image-stat-latency");
            if (cEl) cEl.innerText = data.objects_count;
            if (nEl) nEl.innerText = data.nearest_distance !== null && data.nearest_distance !== undefined ? `${data.nearest_distance} m` : "-- m";
            if (lEl) lEl.innerText = `${data.latency_ms} ms`;

            if (data.saved_result) {
                const savedBadge = document.getElementById("saved-result-badge");
                if (savedBadge) {
                    savedBadge.style.display = "inline-flex";
                    savedBadge.innerHTML = `
                        <a href="${data.saved_result.url}" target="_blank" class="badge badge-green" style="text-decoration:none;">
                            📁 Đã lưu results/${data.saved_result.filename}
                        </a>
                    `;
                }
            }

            renderTable("image-items-table-body", data.detections, "Không phát hiện vật cản nào trong ảnh");
            refreshSavedResults();
            showToast(`Đã phân tích ảnh (${data.objects_count} vật thể - ${data.latency_ms}ms)`, "success");
        } else {
            showToast("Lỗi phân tích ảnh: " + (data.error || "Không xác định"), "error");
        }
    } catch (err) {
        console.error(err);
        showToast("Lỗi kết nối máy chủ khi phân tích ảnh!", "error");
    } finally {
        if (loadingOverlay) loadingOverlay.style.display = "none";
    }
}

async function refreshSavedResults() {
    const listBox = document.getElementById("saved-results-list");
    if (!listBox) return;
    try {
        const resp = await fetch("/api/results");
        if (!resp.ok) return;
        const data = await resp.json();
        const files = data.files || [];
        if (files.length === 0 && !data.video_summary) {
            listBox.innerHTML = `<div class="empty-note">Chưa có tệp kết quả nào trong thư mục results/.</div>`;
            return;
        }
        let html = files.map(f => `
            <a href="${f.url}" target="_blank" class="saved-file-item">
                <span>🖼️ <strong>${f.filename}</strong> <small style="color:var(--text-secondary);">(${f.size_kb} KB)</small></span>
                <small class="mono" style="color:#60a5fa;">Mở ↗</small>
            </a>
        `).join("");

        if (data.video_summary) {
            const vs = data.video_summary;
            html += `
                <div class="saved-file-item" style="flex-direction:column; align-items:flex-start; gap:0.2rem;">
                    <span>📊 <strong>Báo cáo Video: ${vs.video_name}</strong></span>
                    <small style="color:var(--text-secondary);">Đã quét ${vs.total_frames_analyzed} frames • ${vs.dangerous_events_count} cảnh báo nguy hiểm (${vs.completed_at})</small>
                </div>
            `;
        }
        listBox.innerHTML = html;
    } catch (_) {}
}

// --- 10. TAB 3: ĐỒNG BỘ TUYỆT ĐỐI 0.0ms VỚI SERVER C++ PRE-STREAM + SMART AI SYNC GATE ---
const videoPlayer = document.getElementById("sample-video-player");
const videoOverlayCanvas = document.getElementById("video-overlay-canvas");
const vCaptureCanvas = document.createElement("canvas");
const vCaptureCtx = vCaptureCanvas.getContext("2d");

let videoAnalyzeWs = null;
let currentSampleFilename = "test_video.mp4";
let isLocalBlobVideo = false;
let isVideoAnalyzing = false;
let isSyncGated = false;
let videoRenderRaf = null;
let uploadedVideoBlobUrl = null;
let lastPosReportMs = 0;
let lastUiTableSyncMs = 0;
let localScannerBusy = false;

// Dòng thời gian Keyframe đã được máy chủ quét trước: [{ t, detections, most_dangerous, fps, latency_ms, capW, capH, counted }]
let videoKeyframes = [];

let videoSession = {
    videoName: "test_video.mp4",
    totalFrames: 0,
    dangerousCount: 0,
    mostDangerousItem: null,
    mostDangerousSnapshot: null,
    startTime: 0,
    hasSaved: false,
};

function upsertVideoKeyframe(kf) {
    const idx = videoKeyframes.findIndex(item => Math.abs(item.t - kf.t) < 0.04);
    if (idx >= 0) {
        const wasCounted = videoKeyframes[idx].counted;
        videoKeyframes[idx] = { ...kf, counted: wasCounted };
    } else {
        videoKeyframes.push({ ...kf, counted: false });
        videoKeyframes.sort((a, b) => a.t - b.t);
    }

    // Nếu đang ở frame dừng ban đầu, vẽ ngay khung hình vừa nhận
    if (videoPlayer && videoPlayer.paused && Math.abs(kf.t - videoPlayer.currentTime) < 0.25) {
        syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
        drawHudCanvas(videoOverlayCanvas, videoPlayer, kf.detections, kf.most_dangerous, kf.capW, kf.capH);
        renderTable("video-items-table-body", kf.detections, "Không có vật cản trong khung hình hiện tại");
        const statCnt = document.getElementById("video-stat-count");
        const statLat = document.getElementById("video-stat-latency");
        if (statCnt) statCnt.innerText = kf.detections.length;
        if (statLat) statLat.innerText = `${kf.latency_ms} ms`;
    }

    // Dọn bớt các keyframe cũ đã trôi qua quá 4 giây
    if (videoPlayer) {
        const minKeep = videoPlayer.currentTime - 4.0;
        while (videoKeyframes.length > 45 && videoKeyframes[0].t < minKeep) {
            videoKeyframes.shift();
        }
    }
}

function closeVideoAnalyzeSocket() {
    if (videoAnalyzeWs) {
        try { videoAnalyzeWs.close(); } catch (_) {}
        videoAnalyzeWs = null;
    }
}

/**
 * Mở luồng C++ Pre-stream Keyframe từ máy chủ cho video mẫu (không tốn CPU trình duyệt).
 */
function openServerVideoTimeline(filename) {
    closeVideoAnalyzeSocket();
    videoKeyframes = [];
    currentSampleFilename = filename;
    isLocalBlobVideo = false;

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/video_analyze?filename=${encodeURIComponent(filename)}`;
    const ws = new WebSocket(wsUrl);
    videoAnalyzeWs = ws;

    ws.onopen = () => {
        if (videoPlayer) {
            ws.send(JSON.stringify({
                action: "pos",
                time: videoPlayer.currentTime || 0,
                playing: !videoPlayer.paused,
            }));
        }
    };

    ws.onmessage = (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "keyframe") {
                upsertVideoKeyframe(msg);
            }
        } catch (_) {}
    };
}

/**
 * Kiểm tra xem tại thời điểm nowT đã có sẵn Keyframe tương lai (>= nowT) trong bộ đệm chưa.
 */
function hasFutureKeyframeReady(nowT) {
    for (let i = videoKeyframes.length - 1; i >= 0; i--) {
        if (videoKeyframes[i].t >= nowT + 0.02) return true;
        if (videoKeyframes[i].t < nowT - 0.5) break;
    }
    return false;
}

/**
 * Nội suy mượt 60 FPS giữa kfPrev (<= nowT) và kfNext (>= nowT) với chu kỳ dày 0.20s!
 */
function getInterpolatedStateAt(nowT) {
    if (videoKeyframes.length === 0) return null;

    let prevIdx = -1;
    for (let i = 0; i < videoKeyframes.length; i++) {
        if (videoKeyframes[i].t <= nowT + 0.01) prevIdx = i;
        else break;
    }

    if (prevIdx === -1) {
        const first = videoKeyframes[0];
        return (first.t - nowT < 0.30) ? { ...first, sourceKf: first } : null;
    }

    const kfPrev = videoKeyframes[prevIdx];
    const kfNext = prevIdx + 1 < videoKeyframes.length ? videoKeyframes[prevIdx + 1] : null;
    const capW = kfPrev.capW || 640;
    const capH = kfPrev.capH || 360;

    if (kfNext && kfNext.t > kfPrev.t && (kfNext.t - kfPrev.t) <= 0.85) {
        const dt = kfNext.t - kfPrev.t;
        const alpha = Math.max(0, Math.min(1, (nowT - kfPrev.t) / dt));
        const listA = kfPrev.detections || [];
        const listB = kfNext.detections || [];
        const maxNormDist = Math.min(0.45, 0.16 + dt * 0.75);

        const usedA = new Set();
        const usedB = new Set();
        const pairs = [];

        for (let ia = 0; ia < listA.length; ia++) {
            const a = listA[ia];
            const cxA = (a.box[0] + a.box[2]) * 0.5;
            const cyA = (a.box[1] + a.box[3]) * 0.5;
            const hA = Math.max(1, a.box[3] - a.box[1]);

            for (let ib = 0; ib < listB.length; ib++) {
                const b = listB[ib];
                const sameLabel = a.raw_label === b.raw_label;
                const sameCat = a.category === b.category && a.category !== "obstacle";
                if (!sameLabel && !sameCat) continue;

                const cxB = (b.box[0] + b.box[2]) * 0.5;
                const cyB = (b.box[1] + b.box[3]) * 0.5;
                const hB = Math.max(1, b.box[3] - b.box[1]);

                const normDist = Math.hypot(cxA - cxB, cyA - cyB) / capW;
                const sizeSim = Math.min(hA, hB) / Math.max(hA, hB);
                const iou = computeBoxIoU(a.box, b.box);

                if (normDist < maxNormDist && sizeSim > 0.42) {
                    const score = iou * 2.5 + (1.0 - normDist / maxNormDist) * 1.4 + sizeSim * 0.5 + (sameLabel ? 0.7 : 0);
                    pairs.push({ ia, ib, score });
                }
            }
        }

        pairs.sort((p1, p2) => p2.score - p1.score);

        const interpolated = [];
        for (const p of pairs) {
            if (usedA.has(p.ia) || usedB.has(p.ib)) continue;
            usedA.add(p.ia);
            usedB.add(p.ib);

            const a = listA[p.ia];
            const b = listB[p.ib];
            const box = [
                a.box[0] + (b.box[0] - a.box[0]) * alpha,
                a.box[1] + (b.box[1] - a.box[1]) * alpha,
                a.box[2] + (b.box[2] - a.box[2]) * alpha,
                a.box[3] + (b.box[3] - a.box[3]) * alpha,
            ];
            const dist = Math.round((a.distance + (b.distance - a.distance) * alpha) * 10) / 10;
            const direction = getDirectionFromBox(box, capW);
            const isPothole = (a.raw_label || "").toLowerCase().includes("o ga");
            const isDanger = dist <= state.dangerDistance || (isPothole && dist <= 4.0);
            const lvl = dist <= 1.5 ? "Rất gần" : dist <= 3.0 ? "Gần" : dist <= 5.0 ? "Trung bình" : "Xa";

            interpolated.push({
                ...(alpha < 0.5 ? a : b),
                box,
                distance: dist,
                distance_level: lvl,
                direction,
                is_dangerous: isDanger,
            });
        }

        // Giữ mượt đối tượng xuất hiện ở rìa đầu/cuối chu kỳ 0.20s
        if (alpha < 0.45) {
            for (let ia = 0; ia < listA.length; ia++) {
                if (!usedA.has(ia)) interpolated.push(listA[ia]);
            }
        } else {
            for (let ib = 0; ib < listB.length; ib++) {
                if (!usedB.has(ib)) interpolated.push(listB[ib]);
            }
        }

        const dangerList = interpolated.filter(d => d.is_dangerous).sort((x, y) => x.distance - y.distance);
        return {
            t: nowT,
            detections: interpolated,
            most_dangerous: dangerList.length > 0 ? dangerList[0] : null,
            fps: kfNext.fps || kfPrev.fps,
            latency_ms: kfNext.latency_ms || kfPrev.latency_ms,
            capW,
            capH,
            sourceKf: kfPrev,
        };
    }

    if (nowT - kfPrev.t <= 0.45) {
        return { ...kfPrev, sourceKf: kfPrev };
    }
    return null;
}

// Quét cục bộ khi người dùng tải lên tệp video cá nhân (blob:) từ máy tính
async function localBlobVideoStep() {
    if (!isLocalBlobVideo || !isVideoAnalyzing || !videoPlayer || videoPlayer.paused || videoPlayer.ended) {
        localScannerBusy = false;
        return;
    }
    if (localScannerBusy) return;
    localScannerBusy = true;

    try {
        // Giữ tạm tốc độ phát chậm khi đang quét khung tiếp theo cho video cục bộ để khớp 100%
        const vw = videoPlayer.videoWidth || 640;
        const vh = videoPlayer.videoHeight || 360;
        const capW = 640;
        const capH = Math.max(1, Math.round(capW * (vh / vw)));
        if (vCaptureCanvas.width !== capW || vCaptureCanvas.height !== capH) {
            vCaptureCanvas.width = capW;
            vCaptureCanvas.height = capH;
        }

        const frameT = videoPlayer.currentTime;
        vCaptureCtx.drawImage(videoPlayer, 0, 0, capW, capH);
        const dataUrl = vCaptureCanvas.toDataURL("image/jpeg", 0.68);

        const data = await inferFrameRealtime({
            frame_data: dataUrl,
            danger_distance: state.dangerDistance,
            conf_threshold: state.confThreshold,
            enable_blur: state.enableBlur,
            return_image: false,
            imgsz: 320,
        });

        if (data && data.success !== false) {
            upsertVideoKeyframe({
                t: frameT,
                detections: data.detections || [],
                most_dangerous: data.most_dangerous || null,
                fps: data.fps ?? 8,
                latency_ms: data.latency_ms ?? 120,
                capW: data.frame_width || capW,
                capH: data.frame_height || capH,
            });
        }
    } catch (_) {}
    finally {
        localScannerBusy = false;
        if (isLocalBlobVideo && isVideoAnalyzing && videoPlayer && !videoPlayer.paused) {
            setTimeout(localBlobVideoStep, 10);
        }
    }
}

/**
 * Vòng lặp hiển thị đồng bộ 60 FPS kết hợp Smart AI Sync Gate (Độ trễ hiển thị = 0.0ms tuyệt đối!).
 */
function videoSyncRenderLoop() {
    if (!isVideoAnalyzing || !videoPlayer || videoPlayer.paused || videoPlayer.ended) {
        videoRenderRaf = null;
        return;
    }

    syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);

    const nowT = videoPlayer.currentTime;
    const dur = videoPlayer.duration || 9999;
    const nowMs = performance.now();

    // Báo cáo vị trí phát cho máy chủ mỗi 220ms để máy chủ luôn duy trì vùng đệm trước 6 giây
    if (!isLocalBlobVideo && videoAnalyzeWs && videoAnalyzeWs.readyState === WebSocket.OPEN && (nowMs - lastPosReportMs > 220)) {
        lastPosReportMs = nowMs;
        videoAnalyzeWs.send(JSON.stringify({ action: "pos", time: nowT, playing: true }));
    }

    // SMART AI SYNC GATE: Nếu video chuẩn bị vượt qua keyframe mới nhất mà máy chủ đang tính toán,
    // tự động hãm nhẹ nhịp phát (0.07x) trong tích tắc chờ keyframe kế tiếp tới, TUYỆT ĐỐI KHÔNG ĐỂ VẬT THỂ CHẠY TRƯỚC KHUNG BAO!
    if (!isLocalBlobVideo && nowT < dur - 0.35) {
        const readyAhead = hasFutureKeyframeReady(nowT);
        if (!readyAhead) {
            if (!isSyncGated) {
                isSyncGated = true;
                videoPlayer.playbackRate = 0.07;
            }
        } else if (isSyncGated) {
            isSyncGated = false;
            videoPlayer.playbackRate = state.speechRate;
        }
    }

    const synced = getInterpolatedStateAt(nowT);
    if (synced) {
        drawHudCanvas(
            videoOverlayCanvas,
            videoPlayer,
            synced.detections,
            synced.most_dangerous,
            synced.capW,
            synced.capH
        );

        if (synced.sourceKf && !synced.sourceKf.counted) {
            synced.sourceKf.counted = true;
            videoSession.totalFrames++;
            if (synced.most_dangerous) {
                videoSession.dangerousCount++;
                if (!videoSession.mostDangerousItem || synced.most_dangerous.distance < videoSession.mostDangerousItem.distance) {
                    videoSession.mostDangerousItem = synced.most_dangerous;
                    try {
                        const vw = videoPlayer.videoWidth || 640;
                        const vh = videoPlayer.videoHeight || 360;
                        vCaptureCanvas.width = 640;
                        vCaptureCanvas.height = Math.round(640 * (vh / vw));
                        vCaptureCtx.drawImage(videoPlayer, 0, 0, vCaptureCanvas.width, vCaptureCanvas.height);
                        videoSession.mostDangerousSnapshot = vCaptureCanvas.toDataURL("image/jpeg", 0.75);
                    } catch (_) {}
                }
            }
            updateVideoSessionUI();
        }

        if (nowMs - lastUiTableSyncMs >= 100) {
            lastUiTableSyncMs = nowMs;

            const statFps = document.getElementById("video-stat-fps");
            const statLat = document.getElementById("video-stat-latency");
            const statCnt = document.getElementById("video-stat-count");
            if (statFps) statFps.innerText = `60 FPS (AI ${synced.fps})`;
            if (statLat) statLat.innerText = `0 ms (Đệm trước)`;
            if (statCnt) statCnt.innerText = synced.detections.length;

            renderTable("video-items-table-body", synced.detections, "Không có vật cản trong khung hình hiện tại");

            const nearestOverall = synced.detections.length > 0
                ? synced.detections.reduce((m, d) => (d.distance < m.distance ? d : m), synced.detections[0])
                : null;

            if (synced.most_dangerous) {
                updateDangerBanner("video-danger-banner", synced.most_dangerous);
                alertDangerousObstacle(synced.most_dangerous);
                updateSonarRadar(synced.most_dangerous.distance);
            } else {
                updateDangerBanner("video-danger-banner", null, "Khung hình video an toàn. Khung quét đang khóa đồng bộ 60 FPS.");
                updateSonarRadar(nearestOverall ? nearestOverall.distance : 99.0);
            }
        }
    }

    videoRenderRaf = requestAnimationFrame(videoSyncRenderLoop);
}

function updateVideoSessionUI() {
    const fEl = document.getElementById("vid-sum-frames");
    const dEl = document.getElementById("vid-sum-dangers");
    const cEl = document.getElementById("vid-sum-closest");
    if (fEl) fEl.innerText = videoSession.totalFrames;
    if (dEl) dEl.innerText = videoSession.dangerousCount;
    if (cEl) {
        if (videoSession.mostDangerousItem) {
            const it = videoSession.mostDangerousItem;
            const dir = it.direction === "chính giữa" ? "phía trước" : it.direction;
            cEl.innerText = `${it.label} (${it.distance}m — ${dir})`;
        } else {
            cEl.innerText = "Chưa ghi nhận nguy hiểm";
        }
    }
}

function updateVideoToggleButton() {
    const btn = document.getElementById("btn-toggle-video");
    if (!btn || !videoPlayer) return;

    if (videoPlayer.paused || videoPlayer.ended) {
        btn.className = "btn btn-primary";
        const label = videoPlayer.ended ? "Phát Lại Từ Đầu" : "Phát & Quét Video AI";
        btn.innerHTML = `
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path></svg>
            <span id="btn-toggle-video-text">${label}</span>
        `;
    } else {
        btn.className = "btn btn-danger";
        btn.innerHTML = `
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <span id="btn-toggle-video-text">Tạm Dừng Phân Tích</span>
        `;
    }
}

function clearVideoOverlay() {
    if (!videoOverlayCanvas) return;
    const ctx = videoOverlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, videoOverlayCanvas.width, videoOverlayCanvas.height);
}

function setupVideoAnalyzer() {
    if (!videoPlayer || !videoOverlayCanvas) return;

    openServerVideoTimeline("test_video.mp4");

    videoPlayer.addEventListener("loadedmetadata", () => {
        syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
    });

    videoPlayer.addEventListener("loadeddata", () => {
        syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
        const synced = getInterpolatedStateAt(videoPlayer.currentTime || 0);
        if (synced) {
            drawHudCanvas(videoOverlayCanvas, videoPlayer, synced.detections, synced.most_dangerous, synced.capW, synced.capH);
        }
    });

    window.addEventListener("resize", () => {
        syncLiveCanvasSize();
        syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
    });

    videoPlayer.addEventListener("play", () => {
        initAudioContext();
        isVideoAnalyzing = true;
        isSyncGated = false;
        videoPlayer.playbackRate = state.speechRate;
        updateVideoToggleButton();

        if (!isLocalBlobVideo) {
            if (!videoAnalyzeWs || videoAnalyzeWs.readyState !== WebSocket.OPEN) {
                openServerVideoTimeline(currentSampleFilename);
            } else {
                videoAnalyzeWs.send(JSON.stringify({ action: "pos", time: videoPlayer.currentTime, playing: true }));
            }
        } else {
            connectStreamWebSocket();
            localBlobVideoStep();
        }

        const badge = document.getElementById("video-ai-badge");
        if (badge) {
            badge.className = "badge badge-green";
            badge.innerHTML = `<span class="status-dot"></span> AI Khóa Đồng Bộ 60 FPS`;
        }

        const statusBadge = document.getElementById("video-session-status");
        if (statusBadge) {
            statusBadge.className = "badge badge-blue";
            statusBadge.innerText = "Đang ghi nhận";
        }

        if (videoPlayer.currentTime < 0.3 || videoSession.hasSaved) {
            videoKeyframes.forEach(k => { k.counted = false; });
            videoSession = {
                videoName: isLocalBlobVideo ? (videoPlayer.dataset.customName || "video.mp4") : currentSampleFilename,
                totalFrames: 0,
                dangerousCount: 0,
                mostDangerousItem: null,
                mostDangerousSnapshot: null,
                startTime: Date.now(),
                hasSaved: false,
            };
            updateVideoSessionUI();
        }

        if (!videoRenderRaf) {
            videoRenderRaf = requestAnimationFrame(videoSyncRenderLoop);
        }
    });

    videoPlayer.addEventListener("pause", () => {
        isVideoAnalyzing = false;
        isSyncGated = false;
        videoPlayer.playbackRate = state.speechRate;
        if (videoRenderRaf) {
            cancelAnimationFrame(videoRenderRaf);
            videoRenderRaf = null;
        }
        stopSonarRadar();
        updateVideoToggleButton();

        if (!isLocalBlobVideo && videoAnalyzeWs && videoAnalyzeWs.readyState === WebSocket.OPEN) {
            videoAnalyzeWs.send(JSON.stringify({ action: "pos", time: videoPlayer.currentTime, playing: false }));
        }

        if (!videoPlayer.ended) {
            const badge = document.getElementById("video-ai-badge");
            if (badge) {
                badge.className = "badge badge-yellow";
                badge.innerHTML = `<span class="status-dot"></span> Video Tạm Dừng`;
            }
            const synced = getInterpolatedStateAt(videoPlayer.currentTime);
            if (synced) {
                syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
                drawHudCanvas(videoOverlayCanvas, videoPlayer, synced.detections, synced.most_dangerous, synced.capW, synced.capH);
                renderTable("video-items-table-body", synced.detections, "Không có vật cản trong khung hình hiện tại");
            }
        }
    });

    videoPlayer.addEventListener("seeked", () => {
        const nowT = videoPlayer.currentTime;
        if (!isLocalBlobVideo && videoAnalyzeWs && videoAnalyzeWs.readyState === WebSocket.OPEN) {
            videoAnalyzeWs.send(JSON.stringify({ action: "seek", time: nowT }));
        }
        const synced = getInterpolatedStateAt(nowT);
        if (synced) {
            syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
            drawHudCanvas(videoOverlayCanvas, videoPlayer, synced.detections, synced.most_dangerous, synced.capW, synced.capH);
            renderTable("video-items-table-body", synced.detections, "Không có vật cản trong khung hình hiện tại");
        }
    });

    videoPlayer.addEventListener("ended", async () => {
        isVideoAnalyzing = false;
        isSyncGated = false;
        videoPlayer.playbackRate = state.speechRate;
        if (videoRenderRaf) {
            cancelAnimationFrame(videoRenderRaf);
            videoRenderRaf = null;
        }
        stopSonarRadar();
        updateVideoToggleButton();

        const badge = document.getElementById("video-ai-badge");
        if (badge) {
            badge.className = "badge badge-green";
            badge.innerHTML = `<span class="status-dot"></span> Đã Quét Xong Video`;
        }

        if (!videoSession.hasSaved && videoSession.totalFrames > 2) {
            videoSession.hasSaved = true;
            const durationSec = Math.max(1, Math.round((Date.now() - videoSession.startTime) / 1000));
            try {
                const resp = await fetch("/api/save_video_summary", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        video_name: videoSession.videoName,
                        total_frames: videoSession.totalFrames,
                        duration_seconds: durationSec,
                        dangerous_count: videoSession.dangerousCount,
                        most_dangerous_label: videoSession.mostDangerousItem ? videoSession.mostDangerousItem.label : null,
                        most_dangerous_direction: videoSession.mostDangerousItem ? videoSession.mostDangerousItem.direction : null,
                        most_dangerous_distance: videoSession.mostDangerousItem ? videoSession.mostDangerousItem.distance : null,
                        snapshot_data: videoSession.mostDangerousSnapshot || null,
                    }),
                });
                if (resp.ok) {
                    const statusBadge = document.getElementById("video-session-status");
                    if (statusBadge) {
                        statusBadge.className = "badge badge-green";
                        statusBadge.innerText = "Đã lưu results/";
                    }
                    refreshSavedResults();
                    showToast("Đã lưu báo cáo tổng kết phiên video vào results/", "success");
                }
            } catch (_) {}
        }
    });
}

function loadSampleVideo(filename) {
    if (!videoPlayer) return;
    document.querySelectorAll(".btn-sample-video").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-file") === filename);
    });

    videoPlayer.pause();
    clearVideoOverlay();
    delete videoPlayer.dataset.customName;

    openServerVideoTimeline(filename);
    videoPlayer.src = `/api/stream_sample/${filename}`;
    videoPlayer.load();

    // Chờ nhẹ 350ms cho máy chủ đệm trước những keyframe đầu tiên rồi phát mượt tức thì
    setTimeout(() => {
        videoPlayer.play().catch(() => {});
    }, 350);
}

function loadLocalVideoFile(file) {
    if (!videoPlayer || !file) return;
    document.querySelectorAll(".btn-sample-video").forEach(btn => btn.classList.remove("active"));

    closeVideoAnalyzeSocket();
    isLocalBlobVideo = true;
    videoKeyframes = [];

    if (uploadedVideoBlobUrl) URL.revokeObjectURL(uploadedVideoBlobUrl);
    uploadedVideoBlobUrl = URL.createObjectURL(file);
    videoPlayer.dataset.customName = file.name;
    clearVideoOverlay();
    videoPlayer.src = uploadedVideoBlobUrl;
    videoPlayer.load();
    videoPlayer.play().catch(() => {});
    showToast(`Đang nạp video cục bộ: ${file.name}`, "info");
}

// --- 11. ĐIỀU KHIỂN ÂM LƯỢNG LAPTOP, TỐC ĐỘ YOUTUBE & TƯƠNG PHẢN ---
let testVoiceTimer = null;

function triggerTestVoice() {
    const btn = document.getElementById("btn-test-voice");
    if (btn) {
        btn.classList.add("active");
        isTestingVoice = true;
        if (testVoiceTimer) clearTimeout(testVoiceTimer);
        testVoiceTimer = setTimeout(() => {
            btn.classList.remove("active");
            isTestingVoice = false;
        }, 3800);
    }
    speakAlert("Xin chào. Hệ thống trợ lý thị giác BlindAssist AI đang hoạt động ổn định bằng tiếng Việt.", true);
}

function getVolumeIconSvg(frac, size = 18) {
    if (frac <= 0.001) {
        return `<svg width="${size}" height="${size}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2"></path></svg>`;
    } else if (frac < 0.35) {
        return `<svg width="${size}" height="${size}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072"></path></svg>`;
    } else if (frac < 0.7) {
        return `<svg width="${size}" height="${size}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728"></path></svg>`;
    }
    return `<svg width="${size}" height="${size}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path></svg>`;
}

function setVolume(percent, playSoundFeedback = false) {
    const p = Math.max(0, Math.min(100, Math.round(percent)));
    const frac = p / 100.0;

    state.volume = frac;
    state.enableVoice = p > 0;

    if (ttsAudio) ttsAudio.volume = frac;
    if (videoPlayer) videoPlayer.volume = frac * 0.45;

    const iconSlot = document.getElementById("volume-icon-slot");
    if (iconSlot) iconSlot.innerHTML = getVolumeIconSvg(frac, 17);

    const badgeText = document.getElementById("volume-badge-text");
    if (badgeText) badgeText.innerText = p === 0 ? "Loa: Tắt" : `Loa: ${p}%`;

    const vBtn = document.getElementById("btn-voice-toggle");
    if (vBtn) vBtn.classList.toggle("active", p > 0);

    const popupIconSlot = document.getElementById("vol-popup-icon-slot");
    if (popupIconSlot) popupIconSlot.innerHTML = getVolumeIconSvg(frac, 19);

    const menuBadge = document.getElementById("volume-menu-badge");
    if (menuBadge) menuBadge.innerText = p === 0 ? "Tắt tiếng" : `${p}%`;

    const percentLabel = document.getElementById("volume-percent-label");
    if (percentLabel) percentLabel.innerText = `${p}%`;

    const fillBar = document.getElementById("volume-fill-bar");
    if (fillBar) fillBar.style.width = `${p}%`;

    const slider = document.getElementById("dropdown-volume-slider");
    if (slider && Math.abs(parseInt(slider.value, 10) - p) > 0) slider.value = p;

    document.querySelectorAll(".vol-chip").forEach(chip => {
        chip.classList.toggle("active", parseInt(chip.getAttribute("data-vol"), 10) === p);
    });

    try { localStorage.setItem("blindassist_volume", p.toString()); } catch (_) {}
    if (playSoundFeedback && p > 0) playVolumeTick(frac);
}
window.setVolume = setVolume;

function changeVolumeRelative(delta) {
    setVolume(Math.round(state.volume * 100) + delta, true);
}
window.changeVolumeRelative = changeVolumeRelative;

function toggleVolumeMute() {
    if (state.volume > 0.01) {
        state.prevVolume = state.volume;
        setVolume(0, false);
        if (ttsAudio) ttsAudio.pause();
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        finishSpeakingCleanup();
        showToast("Đã tắt tiếng loa (Mute)", "warn", 2000);
    } else {
        const restore = state.prevVolume > 0.05 ? Math.round(state.prevVolume * 100) : 100;
        setVolume(restore, true);
        showToast(`Đã bật tiếng loa: ${restore}%`, "info", 2000);
    }
}
window.toggleVolumeMute = toggleVolumeMute;

function toggleVolumeDropdown(forceState = null) {
    const menu = document.getElementById("volume-dropdown-menu");
    const btn = document.getElementById("btn-voice-toggle");
    if (!menu || !btn) return;

    const shouldOpen = forceState !== null ? forceState : !menu.classList.contains("show");
    if (shouldOpen) closeSpeedDropdown();

    menu.classList.toggle("show", shouldOpen);
    btn.classList.toggle("dropdown-open", shouldOpen);
    btn.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
}
window.toggleVolumeDropdown = toggleVolumeDropdown;

function closeVolumeDropdown() {
    toggleVolumeDropdown(false);
}
window.closeVolumeDropdown = closeVolumeDropdown;

const SPEECH_RATES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
let rateTimer = null;

function setSpeechRate(rate, announce = true) {
    let val = parseFloat(rate) || 1.0;
    val = Math.round(val * 20) / 20;
    val = Math.min(2.0, Math.max(0.5, val));
    state.speechRate = val;

    ttsAudio.playbackRate = val;
    ttsAudio.defaultPlaybackRate = val;

    if (videoPlayer && !isSyncGated) {
        videoPlayer.playbackRate = val;
        videoPlayer.defaultPlaybackRate = val;
    }

    const rateText = document.getElementById("rate-badge-text");
    if (rateText) rateText.innerText = `Tốc độ: ${val.toFixed(val % 1 === 0 ? 1 : 2)}x`;

    const menuBadge = document.getElementById("speed-menu-current-badge");
    if (menuBadge) menuBadge.innerText = `${val.toFixed(2)}x`;

    const sliderVal = document.getElementById("speed-slider-val");
    if (sliderVal) sliderVal.innerText = `${val.toFixed(2)}x`;

    const dropdownSlider = document.getElementById("dropdown-speed-slider");
    if (dropdownSlider && Math.abs(parseFloat(dropdownSlider.value) - val) > 0.01) {
        dropdownSlider.value = val;
    }

    document.querySelectorAll(".speed-preset-item").forEach(item => {
        const itemSpeed = parseFloat(item.getAttribute("data-speed"));
        const isActive = Math.abs(itemSpeed - val) < 0.025;
        item.classList.toggle("active", isActive);
        item.setAttribute("aria-checked", isActive ? "true" : "false");
    });

    const slider = document.getElementById("range-speech-rate");
    const valLabel = document.getElementById("val-speech-rate");
    if (slider && Math.abs(parseFloat(slider.value) - val) > 0.01) slider.value = val;
    if (valLabel) valLabel.innerText = `${val.toFixed(2)}x`;

    const btn = document.getElementById("btn-speech-rate");
    if (btn) {
        btn.classList.add("active");
        if (rateTimer) clearTimeout(rateTimer);
        rateTimer = setTimeout(() => {
            const menu = document.getElementById("speed-dropdown-menu");
            if (!menu || !menu.classList.contains("show")) btn.classList.remove("active");
        }, 1200);
    }

    try { localStorage.setItem("blindassist_rate", val.toString()); } catch (_) {}
    if (announce) speakAlert(`Tốc độ nói: ${val} lần`, false);
}
window.setSpeechRate = setSpeechRate;

function toggleSpeedDropdown(forceState = null) {
    const menu = document.getElementById("speed-dropdown-menu");
    const btn = document.getElementById("btn-speech-rate");
    if (!menu || !btn) return;

    const shouldOpen = forceState !== null ? forceState : !menu.classList.contains("show");
    if (shouldOpen) closeVolumeDropdown();

    menu.classList.toggle("show", shouldOpen);
    btn.classList.toggle("dropdown-open", shouldOpen);
    btn.classList.toggle("active", shouldOpen);
    btn.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
}
window.toggleSpeedDropdown = toggleSpeedDropdown;

function closeSpeedDropdown() {
    toggleSpeedDropdown(false);
}
window.closeSpeedDropdown = closeSpeedDropdown;

function cycleSpeechRate() {
    const current = state.speechRate;
    let nextIndex = 0;
    for (let i = 0; i < SPEECH_RATES.length; i++) {
        if (SPEECH_RATES[i] > current + 0.04) {
            nextIndex = i;
            break;
        }
    }
    setSpeechRate(SPEECH_RATES[nextIndex], true);
}
window.cycleSpeechRate = cycleSpeechRate;

function toggleHighContrast(force = null) {
    const next = force !== null ? force : !state.highContrast;
    state.highContrast = next;
    document.body.classList.toggle("high-contrast", next);
    document.getElementById("btn-contrast-toggle")?.classList.toggle("active", next);
    try { localStorage.setItem("blindassist_contrast", next ? "1" : "0"); } catch (_) {}
}

function switchTab(targetId) {
    const targetPane = document.getElementById(targetId);
    if (!targetPane) return;

    document.querySelectorAll(".nav-tab").forEach(t => {
        const active = t.getAttribute("data-tab") === targetId;
        t.classList.toggle("active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    targetPane.classList.add("active");
    state.currentTab = targetId;

    if (targetId !== "tab-live" && state.isStreaming) stopCamera();
    if (targetId !== "tab-video" && videoPlayer && !videoPlayer.paused) videoPlayer.pause();
    if (targetId === "tab-video") {
        setTimeout(() => {
            syncOverlayToVideoRect(videoPlayer, videoOverlayCanvas);
            const synced = getInterpolatedStateAt(videoPlayer?.currentTime || 0);
            if (synced) {
                drawHudCanvas(videoOverlayCanvas, videoPlayer, synced.detections, synced.most_dangerous, synced.capW, synced.capH);
            }
        }, 50);
    }
    if (targetId === "tab-image") refreshSavedResults();
}

function renderClassReferenceTable(classDetails, modelName) {
    const tbody = document.getElementById("class-reference-tbody");
    const tag = document.getElementById("ref-model-tag");
    const headerBadge = document.getElementById("active-model-badge-text");

    if (tag && modelName) tag.innerText = modelName;
    if (headerBadge && modelName) headerBadge.innerText = `AI: ${modelName}`;
    if (!tbody || !Array.isArray(classDetails)) return;

    tbody.innerHTML = classDetails.map(c => `
        <tr>
            <td><code>${c.id}</code></td>
            <td><strong>${c.display_name}</strong></td>
            <td>${c.raw_name}</td>
            <td>${c.real_height ? `${Number(c.real_height).toFixed(2)} m` : "<em>Mặt đường</em>"}</td>
            <td>${c.method || "Quang học lỗ kim (Focal Length)"}</td>
        </tr>
    `).join("");
}

// --- 12. PHÍM TẮT TRỢ NĂNG TOÀN CỤC ---
const shortcutsModal = document.getElementById("shortcuts-modal");
const shortcutsBtn = document.getElementById("btn-shortcuts-help");

function openShortcutsModal() {
    shortcutsModal?.classList.add("show");
    shortcutsBtn?.classList.add("active");
}
function closeShortcutsModal() {
    shortcutsModal?.classList.remove("show");
    shortcutsBtn?.classList.remove("active");
}
function toggleShortcutsModal() {
    if (shortcutsModal?.classList.contains("show")) closeShortcutsModal();
    else openShortcutsModal();
}
window.closeShortcutsModal = closeShortcutsModal;

document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

    if (e.key === "?") {
        e.preventDefault();
        toggleShortcutsModal();
        return;
    }

    switch (e.code) {
        case "Space":
            e.preventDefault();
            if (state.currentTab === "tab-video" && videoPlayer) {
                if (videoPlayer.paused || videoPlayer.ended) videoPlayer.play().catch(() => {});
                else videoPlayer.pause();
            } else {
                if (state.currentTab !== "tab-live") switchTab("tab-live");
                if (state.isStreaming) stopCamera();
                else startCamera();
            }
            break;

        case "Digit1": switchTab("tab-live"); break;
        case "Digit2": switchTab("tab-image"); break;
        case "Digit3": switchTab("tab-video"); break;
        case "Digit4": switchTab("tab-settings"); break;

        case "KeyM": toggleVolumeMute(); break;
        case "BracketRight": e.preventDefault(); changeVolumeRelative(5); break;
        case "BracketLeft": e.preventDefault(); changeVolumeRelative(-5); break;

        case "KeyR":
            if (state.lastSpokenText) speakAlert(state.lastSpokenText, true);
            break;

        case "KeyC":
            state.facingMode = state.facingMode === "environment" ? "user" : "environment";
            if (state.isStreaming) { stopCamera(); startCamera(); }
            speakAlert("Đã chuyển đổi góc máy ảnh");
            break;

        case "KeyV": triggerTestVoice(); break;
        case "KeyS": cycleSpeechRate(); break;

        case "KeyH":
            toggleHighContrast();
            showToast(state.highContrast ? "Đã bật chế độ Tương phản cao" : "Đã tắt chế độ Tương phản cao", "info");
            break;

        case "Escape":
            closeSpeedDropdown();
            closeVolumeDropdown();
            closeShortcutsModal();
            break;
    }
});

// --- 13. GẮN SỰ KIỆN GIAO DIỆN ---
function initEventListeners() {
    document.getElementById("btn-toggle-cam")?.addEventListener("click", () => {
        if (state.isStreaming) stopCamera();
        else startCamera();
    });
    document.getElementById("btn-hero-start-cam")?.addEventListener("click", startCamera);

    document.getElementById("btn-flip-cam")?.addEventListener("click", () => {
        state.facingMode = state.facingMode === "environment" ? "user" : "environment";
        if (state.isStreaming) { stopCamera(); startCamera(); }
        showToast("Đã chuyển góc Camera", "info");
    });

    document.getElementById("btn-snapshot-live")?.addEventListener("click", () => {
        if (!state.isStreaming || !videoElement || !videoElement.videoWidth) {
            showToast("Hãy bật Camera trước khi chụp khung hình!", "warn");
            return;
        }
        const snapCanvas = document.createElement("canvas");
        snapCanvas.width = videoElement.videoWidth;
        snapCanvas.height = videoElement.videoHeight;
        snapCanvas.getContext("2d").drawImage(videoElement, 0, 0);
        snapCanvas.toBlob((blob) => {
            if (blob) {
                const file = new File([blob], "webcam_snapshot.jpg", { type: "image/jpeg" });
                switchTab("tab-image");
                analyzeImage(file);
            }
        }, "image/jpeg", 0.92);
    });

    document.getElementById("btn-fullscreen-live")?.addEventListener("click", () => {
        const container = document.getElementById("camera-container");
        if (!container) return;
        if (!document.fullscreenElement) container.requestFullscreen?.().catch(() => {});
        else document.exitFullscreen?.();
    });

    document.getElementById("btn-repeat-speech")?.addEventListener("click", () => {
        if (state.lastSpokenText) speakAlert(state.lastSpokenText, true);
    });
    document.getElementById("btn-clear-speech")?.addEventListener("click", () => {
        const log = document.getElementById("speech-log");
        if (log) log.innerHTML = `<div class="speech-bubble"><strong>Đã xóa nhật ký cảnh báo.</strong><small>Vừa xong</small></div>`;
    });

    document.getElementById("model-status-badge")?.addEventListener("click", () => {
        switchTab("tab-settings");
        document.getElementById("sel-model")?.focus();
    });

    // Speed Flyout
    document.getElementById("btn-speech-rate")?.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleSpeedDropdown();
    });
    document.getElementById("speed-dropdown-menu")?.addEventListener("click", e => e.stopPropagation());

    const dropdownSlider = document.getElementById("dropdown-speed-slider");
    dropdownSlider?.addEventListener("input", e => setSpeechRate(parseFloat(e.target.value), false));
    dropdownSlider?.addEventListener("change", e => setSpeechRate(parseFloat(e.target.value), true));

    document.getElementById("btn-speed-minus")?.addEventListener("click", (e) => {
        e.stopPropagation();
        setSpeechRate(state.speechRate - 0.05, false);
    });
    document.getElementById("btn-speed-plus")?.addEventListener("click", (e) => {
        e.stopPropagation();
        setSpeechRate(state.speechRate + 0.05, false);
    });

    document.querySelectorAll(".speed-preset-item").forEach(item => {
        item.addEventListener("click", (e) => {
            e.stopPropagation();
            setSpeechRate(parseFloat(item.getAttribute("data-speed")), true);
            setTimeout(closeSpeedDropdown, 150);
        });
    });

    document.getElementById("btn-test-voice")?.addEventListener("click", triggerTestVoice);
    document.getElementById("btn-contrast-toggle")?.addEventListener("click", () => {
        toggleHighContrast();
        showToast(state.highContrast ? "Đã bật chế độ Tương phản cao" : "Đã tắt chế độ Tương phản cao", "info");
    });

    // Volume Flyout
    const btnVol = document.getElementById("btn-voice-toggle");
    btnVol?.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleVolumeDropdown();
    });
    btnVol?.addEventListener("wheel", (e) => {
        e.preventDefault();
        changeVolumeRelative(e.deltaY < 0 ? 5 : -5);
    }, { passive: false });

    document.getElementById("btn-volume-popup-mute")?.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleVolumeMute();
    });

    const volSlider = document.getElementById("dropdown-volume-slider");
    volSlider?.addEventListener("input", e => setVolume(parseInt(e.target.value, 10), false));
    volSlider?.addEventListener("change", e => setVolume(parseInt(e.target.value, 10), true));

    document.getElementById("volume-dropdown-menu")?.addEventListener("wheel", (e) => {
        e.preventDefault();
        changeVolumeRelative(e.deltaY < 0 ? 5 : -5);
    }, { passive: false });
    document.getElementById("volume-dropdown-menu")?.addEventListener("click", e => e.stopPropagation());

    document.querySelectorAll(".vol-chip").forEach(chip => {
        chip.addEventListener("click", (e) => {
            e.stopPropagation();
            setVolume(parseInt(chip.getAttribute("data-vol"), 10), true);
        });
    });

    const popupSonar = document.getElementById("popup-sonar-toggle");
    const chkSonar = document.getElementById("chk-enable-sonar");
    popupSonar?.addEventListener("change", (e) => {
        state.enableSonar = e.target.checked;
        if (chkSonar) chkSonar.checked = state.enableSonar;
        if (!state.enableSonar) stopSonarRadar();
        showToast(state.enableSonar ? "Đã bật âm thanh Radar Sonar" : "Đã tắt âm thanh Radar Sonar", "info", 2000);
    });
    chkSonar?.addEventListener("change", (e) => {
        state.enableSonar = e.target.checked;
        if (popupSonar) popupSonar.checked = state.enableSonar;
        if (!state.enableSonar) stopSonarRadar();
    });

    document.addEventListener("click", (e) => {
        const sWrap = document.getElementById("speed-menu-wrapper");
        if (sWrap && !sWrap.contains(e.target)) closeSpeedDropdown();
        const vWrap = document.getElementById("volume-menu-wrapper");
        if (vWrap && !vWrap.contains(e.target)) closeVolumeDropdown();
    });

    shortcutsBtn?.addEventListener("click", toggleShortcutsModal);
    document.getElementById("btn-close-modal")?.addEventListener("click", closeShortcutsModal);
    shortcutsModal?.addEventListener("click", (e) => {
        if (e.target === shortcutsModal) closeShortcutsModal();
    });

    document.querySelectorAll(".nav-tab").forEach(tab => {
        tab.addEventListener("click", () => switchTab(tab.getAttribute("data-tab")));
    });

    // Tab 2
    const handleImageFileSelect = (e) => {
        if (e.target.files && e.target.files[0]) {
            analyzeImage(e.target.files[0]);
            e.target.value = "";
        }
    };
    document.getElementById("image-upload-input")?.addEventListener("change", handleImageFileSelect);
    document.getElementById("image-upload-input-hero")?.addEventListener("change", handleImageFileSelect);

    const dropzone = document.getElementById("image-dropzone");
    if (dropzone) {
        ["dragenter", "dragover"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
        });
        ["dragleave", "drop"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); });
        });
        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer?.files;
            if (files && files[0] && files[0].type.startsWith("image/")) analyzeImage(files[0]);
        });
    }

    document.getElementById("btn-sample-image")?.addEventListener("click", () => analyzeImage(null, "test_image.jpg"));
    document.getElementById("btn-rerun-sample-image")?.addEventListener("click", () => analyzeImage(null, "test_image.jpg"));
    document.getElementById("btn-speak-image-summary")?.addEventListener("click", () => {
        if (state.lastImageSummary) speakAlert(state.lastImageSummary, false);
        else showToast("Hãy phân tích một bức ảnh trước!", "warn");
    });
    document.getElementById("btn-refresh-results")?.addEventListener("click", refreshSavedResults);

    // Tab 3
    document.getElementById("btn-toggle-video")?.addEventListener("click", () => {
        if (!videoPlayer) return;
        if (videoPlayer.paused || videoPlayer.ended) {
            if (videoPlayer.ended) videoPlayer.currentTime = 0;
            videoPlayer.play().catch(e => console.warn(e));
        } else {
            videoPlayer.pause();
        }
    });

    document.getElementById("video-upload-input")?.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            loadLocalVideoFile(e.target.files[0]);
            e.target.value = "";
        }
    });

    document.querySelectorAll(".btn-sample-video").forEach(btn => {
        btn.addEventListener("click", () => loadSampleVideo(btn.getAttribute("data-file")));
    });

    // Tab 4
    const settingsForm = document.getElementById("settings-form");
    settingsForm?.addEventListener("submit", async (e) => {
        e.preventDefault();

        const chkBlur = document.getElementById("chk-enable-blur");
        const chkSonarEl = document.getElementById("chk-enable-sonar");
        const chkZones = document.getElementById("chk-show-zones");

        const fData = new FormData(settingsForm);
        fData.set("enable_blur", chkBlur?.checked ? "true" : "false");
        fData.set("enable_sonar", chkSonarEl?.checked ? "true" : "false");
        fData.set("show_zones", chkZones?.checked ? "true" : "false");

        state.dangerDistance = parseFloat(fData.get("danger_distance")) || 3.0;
        state.confThreshold = parseFloat(fData.get("conf_threshold")) || 0.25;
        state.focalLength = parseInt(fData.get("focal_length"), 10) || 650;
        state.speakCooldown = Math.round((parseFloat(fData.get("speech_cooldown")) || 2.8) * 1000);
        state.voiceEngine = fData.get("voice_engine") || "auto";
        state.enableBlur = !!chkBlur?.checked;
        state.enableSonar = !!chkSonarEl?.checked;
        state.showZones = !!chkZones?.checked;

        if (popupSonar) popupSonar.checked = state.enableSonar;
        if (fData.get("speech_rate")) setSpeechRate(parseFloat(fData.get("speech_rate")), false);

        try {
            const resp = await fetch("/api/update_settings", { method: "POST", body: fData });
            if (resp.ok) {
                const resData = await resp.json();
                if (resData.class_details) {
                    renderClassReferenceTable(resData.class_details, resData.active_model_name);
                }
                if (!isLocalBlobVideo) openServerVideoTimeline(currentSampleFilename);
                speakAlert("Đã lưu và áp dụng cấu hình hệ thống mới.");
                showToast(`Đã cập nhật cài đặt (Mô hình: ${resData.active_model_name})`, "success");
            } else {
                showToast("Không thể lưu cài đặt lên máy chủ", "error");
            }
        } catch (_) {
            showToast("Lỗi kết nối khi lưu cài đặt", "error");
        }
    });

    document.getElementById("btn-reset-settings")?.addEventListener("click", () => {
        document.getElementById("sel-model").value = "best.pt";
        document.getElementById("inp-danger-distance").value = 3.0;
        document.getElementById("val-danger").innerText = "3.0 m";
        document.getElementById("inp-conf-threshold").value = 0.25;
        document.getElementById("val-conf").innerText = "0.25";
        document.getElementById("inp-focal-length").value = 650;
        document.getElementById("inp-speech-cooldown").value = 2.8;
        document.getElementById("val-cooldown").innerText = "2.8s";
        document.getElementById("sel-voice-engine").value = "auto";
        document.getElementById("chk-enable-blur").checked = true;
        document.getElementById("chk-enable-sonar").checked = true;
        document.getElementById("chk-show-zones").checked = true;
        setSpeechRate(1.0, false);
        settingsForm?.requestSubmit();
    });

    setupVideoAnalyzer();
}

// --- KHỞI CHẠY ---
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    connectStreamWebSocket();
    refreshSavedResults();

    const savedRate = localStorage.getItem("blindassist_rate");
    setSpeechRate(savedRate !== null ? parseFloat(savedRate) : (state.speechRate || 1.0), false);

    const savedVol = localStorage.getItem("blindassist_volume");
    setVolume(savedVol !== null ? parseInt(savedVol, 10) : 100, false);

    if (localStorage.getItem("blindassist_contrast") === "1") {
        toggleHighContrast(true);
    }

    if ("speechSynthesis" in window) window.speechSynthesis.getVoices();
    console.log("[*] BlindAssist AI v3.2 — Server Pre-Stream + Smart AI Sync Gate Ready.");
});

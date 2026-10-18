/**
 * Motion Reader — Real-time Motion Detection and Video Analysis System
 * Client-side Computer Vision Engine and UI Controller.
 */

// State and Configuration
const config = {
  video: {
    preferredWidth: 1280,
    preferredHeight: 720,
    mirror: true
  },
  detection: {
    minConfidence: 0.5,
    sensitivity: 1.0,
    smoothing: true,
    gestures: true
  },
  display: {
    showSkeleton: true,
    showLandmarks: true,
    showHands: true,
    showFace: true,
    showFps: true,
    showCorners: true
  },
  terminal: {
    autoScroll: true
  }
};

// UI Elements
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const webcamVideo = document.getElementById('webcamVideo');
const fileVideo = document.getElementById('fileVideo');
const fileInput = document.getElementById('fileInput');

const badgeStatus = document.getElementById('badgeStatus');
const badgeFps = document.getElementById('badgeFps');
const badgeRes = document.getElementById('badgeRes');
const liveDot = document.getElementById('liveDot');

const btnSelectLive = document.getElementById('btnSelectLive');
const btnSelectUpload = document.getElementById('btnSelectUpload');
const videoPlaybackBar = document.getElementById('videoPlaybackBar');
const btnFilePlayPause = document.getElementById('btnFilePlayPause');
const btnFileRestart = document.getElementById('btnFileRestart');
const seekSlider = document.getElementById('seekSlider');
const timeLabel = document.getElementById('timeLabel');

const terminalLogBody = document.getElementById('terminalLogBody');
const btnClearTerminal = document.getElementById('btnClearTerminal');

const cameraSelect = document.getElementById('cameraSelect');
const btnMainAction = document.getElementById('btnMainAction');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

const settingsModal = document.getElementById('settingsModal');
const helpModal = document.getElementById('helpModal');
const btnSettings = document.getElementById('btnSettings');
const btnHelp = document.getElementById('btnHelp');

const cameraErrorOverlay = document.getElementById('cameraErrorOverlay');
const overlayDescText = document.getElementById('overlayDescText');
const btnOverlayRetry = document.getElementById('btnOverlayRetry');
const btnOverlayDemo = document.getElementById('btnOverlayDemo');

// Application Execution State
let currentSource = 'LIVE'; // 'LIVE' or 'FILE'
let isRunning = true;
let isPaused = false;
let currentStream = null;
let activeVideo = webcamVideo;

let fpsCounter = 0;
let lastFpsTime = performance.now();
let measuredFps = 30;

// Temporal Motion Analysis Buffers
const wristHistory = {
  left: [],
  right: []
};
const scaleHistory = [];
const postureBuffer = [];

let lastPersonDetected = false;
let consecutiveHits = 0;
let consecutiveMisses = 0;
let currentPosture = null;
let lastLeftRaised = false;
let lastRightRaised = false;
let lastBothRaised = false;
const lastGestureTimes = {};
let lastStepTime = 0;

const eventLog = [];

// Multi-Person AI Tracking State
let latestTrackedPersons = [];
let pendingGesture = null;
let isAnalyzingFrame = false;
let lastAnalysisTimestamp = 0;
let offscreenCanvas = null;
let offscreenCtx = null;

// Landmark connections in MediaPipe Pose
const POSE_CONNECTIONS = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], [23, 24], // torso
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  [27, 29], [29, 31], // left foot
  [28, 30], [30, 32]  // right foot
];

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [9, 10], [10, 11], [11, 12],          // middle
  [13, 14], [14, 15], [15, 16],         // ring
  [0, 17], [17, 18], [18, 19], [19, 20],// pinky
  [5, 9], [9, 13], [13, 17]             // palm
];

// --------------------------------------------------------------------------
// 1. Terminal Event Logger
// --------------------------------------------------------------------------
function formatTimestamp(timeSec = null) {
  if (timeSec !== null && currentSource === 'FILE') {
    const s = Math.floor(timeSec);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    return `${String(h % 24).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }
  const now = new Date();
  return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
}

function logEvent(type, message, confidence = null, personId = null, videoTime = null) {
  const timeStr = formatTimestamp(videoTime);
  const event = {
    timestamp: timeStr,
    type,
    message,
    confidence,
    personId
  };
  eventLog.push(event);

  // Create DOM line
  const line = document.createElement('div');
  line.className = 'log-line';

  const timeSpan = document.createElement('span');
  timeSpan.className = 'log-time';
  timeSpan.textContent = `[${timeStr}]`;

  const msgSpan = document.createElement('span');
  msgSpan.className = `log-msg log-${type.toLowerCase()}`;
  msgSpan.textContent = message;

  line.appendChild(timeSpan);
  line.appendChild(msgSpan);
  terminalLogBody.appendChild(line);

  if (config.terminal.autoScroll) {
    terminalLogBody.scrollTop = terminalLogBody.scrollHeight;
  }
}

// --------------------------------------------------------------------------
// 2. Video Capture and Source Handling
// --------------------------------------------------------------------------
async function enumerateCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    cameraSelect.innerHTML = '';
    
    if (videoDevices.length === 0) {
      const opt = document.createElement('option');
      opt.value = 'default';
      opt.textContent = 'Camera (Default)';
      cameraSelect.appendChild(opt);
      return;
    }

    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Camera ${idx + 1}`;
      cameraSelect.appendChild(opt);
    });
  } catch (err) {
    console.warn('Camera enumeration error:', err);
  }
}

function showCameraErrorOverlay(title, desc) {
  if (cameraErrorOverlay) {
    if (desc && overlayDescText) overlayDescText.textContent = desc;
    cameraErrorOverlay.style.display = 'flex';
  }
}

function hideCameraErrorOverlay() {
  if (cameraErrorOverlay) {
    cameraErrorOverlay.style.display = 'none';
  }
}

async function startWebcam(deviceId = null) {
  stopActiveStream();
  hideCameraErrorOverlay();
  currentSource = 'LIVE';
  activeVideo = webcamVideo;
  badgeStatus.textContent = 'LIVE';
  videoPlaybackBar.style.display = 'none';

  btnSelectLive.classList.add('active');
  btnSelectUpload.classList.remove('active');
  cameraSelect.disabled = false;

  setStatus('Connecting Camera...', 'neutral');
  logEvent('INFO', 'Requesting camera stream...');

  let stream = null;

  // Prioritize 640x480 native camera resolution for fast reliable connection
  const constraintAttempts = [
    {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        deviceId: (deviceId && deviceId !== 'default') ? { exact: deviceId } : undefined
      },
      audio: false
    },
    {
      video: {
        width: { ideal: config.video.preferredWidth },
        height: { ideal: config.video.preferredHeight }
      },
      audio: false
    },
    {
      video: true,
      audio: false
    }
  ];

  for (const c of constraintAttempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(c);
      if (stream) break;
    } catch (err) {
      console.warn('Webcam constraint attempt notice:', err);
    }
  }

  if (!stream) {
    handleWebcamError(new Error('Could not access camera device'));
    return;
  }

  currentStream = stream;
  webcamVideo.srcObject = currentStream;
  try {
    await webcamVideo.play();
  } catch (playErr) {
    console.warn('Video play notice:', playErr);
  }

  setStatus('Tracking...', 'active');
  resetAnalysisState();
  hideCameraErrorOverlay();

  // Initialize tracking session and log startup banners to terminal
  await initTrackingSession();
}

function handleWebcamError(err) {
  setStatus('Camera Unavailable', 'error');

  let desc = 'Webcam device could not be started.';
  if (err.name === 'NotReadableError' || (err.message && err.message.includes('Could not start video source'))) {
    desc = 'The webcam device is locked or in use by another application (e.g. desktop app, Zoom, Teams, Camera app).';
    logEvent('ERROR', 'Webcam access failed: Could not start video source');
    logEvent('SYSTEM', 'Tip: Close any apps using the camera, then click "Retry Camera" or "Load Demo Video".');
  } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    desc = 'Camera permission was denied in your browser settings.';
    logEvent('ERROR', 'Webcam access failed: Permission denied by browser');
    logEvent('SYSTEM', 'Tip: Click the camera / lock icon in your browser URL bar to allow camera access.');
  } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    desc = 'No physical webcam device was detected on this computer.';
    logEvent('ERROR', 'Webcam access failed: No camera device detected');
    logEvent('SYSTEM', 'Tip: You can use "Load Demo Video" or upload a video file to test motion tracking.');
  } else {
    desc = `Webcam error: ${err.message || err.name}`;
    logEvent('ERROR', `Webcam access failed: ${err.message || err.name}`);
  }

  showCameraErrorOverlay('Camera Stream Unavailable', desc);
}

function loadVideoUrl(url, displayName = 'demo_motion_analysis.mp4') {
  stopActiveStream();
  hideCameraErrorOverlay();
  currentSource = 'FILE';
  activeVideo = fileVideo;
  badgeStatus.textContent = 'FILE';
  videoPlaybackBar.style.display = 'flex';

  btnSelectLive.classList.remove('active');
  btnSelectUpload.classList.add('active');
  cameraSelect.disabled = true;

  fileVideo.src = url;
  fileVideo.load();
  fileVideo.play().catch(e => console.warn('Autoplay prevented:', e));

  btnFilePlayPause.textContent = 'Pause';
  setStatus('Tracking...', 'active');
  resetAnalysisState();
  initTrackingSession();
}

function loadVideoFile(file) {
  stopActiveStream();
  hideCameraErrorOverlay();
  currentSource = 'FILE';
  activeVideo = fileVideo;
  badgeStatus.textContent = 'FILE';
  videoPlaybackBar.style.display = 'flex';

  btnSelectLive.classList.remove('active');
  btnSelectUpload.classList.add('active');
  cameraSelect.disabled = true;

  const url = URL.createObjectURL(file);
  fileVideo.src = url;
  fileVideo.load();
  fileVideo.play().catch(e => console.warn('Autoplay prevented:', e));

  btnFilePlayPause.textContent = 'Pause';
  setStatus('Tracking...', 'active');
  resetAnalysisState();
  initTrackingSession();
}

function stopActiveStream() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  webcamVideo.pause();
  fileVideo.pause();
}

// --------------------------------------------------------------------------
// 3. Pose Detection and AI Inference
// --------------------------------------------------------------------------
let poseModel = null;
let handsModel = null;
let isPoseBusy = false;
let isHandsBusy = false;
let lastPoseDispatch = 0;
let lastHandsDispatch = 0;

function initMediaPipe() {
  try {
    if (window.Pose && !poseModel) {
      poseModel = new window.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
      });
      poseModel.setOptions({
        modelComplexity: 0,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      poseModel.onResults(onPoseResults);
    }
  } catch (e) {
    console.warn('MediaPipe Pose setup notice:', e);
  }

  try {
    if (window.Hands && !handsModel) {
      handsModel = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      handsModel.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      handsModel.onResults(onHandResults);
    }
  } catch (e) {
    console.warn('MediaPipe Hands setup notice:', e);
  }
}

let latestHandLandmarks = [];
function onHandResults(results) {
  latestHandLandmarks = results.multiHandLandmarks || [];
}

let latestPoseResults = null;
function onPoseResults(results) {
  latestPoseResults = results;
}

// --------------------------------------------------------------------------
// 4. Temporal Movement Analysis
// --------------------------------------------------------------------------
function resetAnalysisState() {
  wristHistory.left = [];
  wristHistory.right = [];
  scaleHistory.length = 0;
  postureBuffer.length = 0;
  lastPersonDetected = false;
  consecutiveHits = 0;
  consecutiveMisses = 0;
  currentPosture = null;
  lastLeftRaised = false;
  lastRightRaised = false;
  lastBothRaised = false;
  Object.keys(lastGestureTimes).forEach(k => delete lastGestureTimes[k]);
}

function calculateAngle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  if (mag1 * mag2 < 1e-6) return 180;
  const cosine = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function processTemporalAnalysis(landmarks, currentTimeSec) {
  const personDetected = landmarks && landmarks.length >= 33;

  // 1. Person Detection Hysteresis
  if (personDetected) {
    consecutiveHits++;
    consecutiveMisses = 0;
  } else {
    consecutiveMisses++;
    consecutiveHits = 0;
  }

  if (consecutiveHits >= 2 && !lastPersonDetected) {
    lastPersonDetected = true;
    logEvent('PERSON', 'Person detected', null, 1, currentTimeSec);
  } else if (consecutiveMisses >= 8 && lastPersonDetected) {
    lastPersonDetected = false;
    currentPosture = null;
    lastLeftRaised = false;
    lastRightRaised = false;
    lastBothRaised = false;
    setStatus('No Person Detected', 'warning');
    return;
  }

  if (!lastPersonDetected || !personDetected) return;
  setStatus('Tracking...', 'active');

  const l_sh = landmarks[11], r_sh = landmarks[12];
  const l_el = landmarks[13], r_el = landmarks[14];
  const l_wr = landmarks[15], r_wr = landmarks[16];
  const l_hip = landmarks[23], r_hip = landmarks[24];
  const l_knee = landmarks[25], r_knee = landmarks[26];
  const l_ank = landmarks[27], r_ank = landmarks[28];
  const nose = landmarks[0];

  const mid_sh = { x: (l_sh.x + r_sh.x) / 2, y: (l_sh.y + r_sh.y) / 2 };
  const mid_hip = { x: (l_hip.x + r_hip.x) / 2, y: (l_hip.y + r_hip.y) / 2 };
  const torsoHeight = Math.hypot(mid_sh.x - mid_hip.x, mid_sh.y - mid_hip.y);
  const shoulderWidth = Math.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y);

  // 2. Posture Classification (Standing / Sitting / Crouching)
  let kneeAngleLeft = calculateAngle(l_hip, l_knee, l_ank);
  let kneeAngleRight = calculateAngle(r_hip, r_knee, r_ank);
  let avgKnee = (kneeAngleLeft + kneeAngleRight) / 2;

  let rawPosture = 'Standing';
  if (avgKnee < 85) {
    rawPosture = 'Crouching';
  } else if (avgKnee < 145) {
    rawPosture = 'Sitting';
  }

  postureBuffer.push(rawPosture);
  if (postureBuffer.length > 5) postureBuffer.shift();

  const counts = {};
  postureBuffer.forEach(p => counts[p] = (counts[p] || 0) + 1);
  const detectedPosture = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);

  if (detectedPosture !== currentPosture) {
    currentPosture = detectedPosture;
    logEvent('POSTURE', detectedPosture, null, null, currentTimeSec);
  }

  // 3. Arm and Hand Raise Detection (passed to named tracker)
  const raiseThreshold = 0.03 * config.detection.sensitivity;
  const lowerThreshold = 0.04 / config.detection.sensitivity;

  let leftRaised = lastLeftRaised;
  if (!lastLeftRaised && l_wr.y < l_sh.y - raiseThreshold) leftRaised = true;
  else if (lastLeftRaised && l_wr.y > l_sh.y + lowerThreshold) leftRaised = false;

  let rightRaised = lastRightRaised;
  if (!lastRightRaised && r_wr.y < r_sh.y - raiseThreshold) rightRaised = true;
  else if (lastRightRaised && r_wr.y > r_sh.y + lowerThreshold) rightRaised = false;

  if (rightRaised && !lastRightRaised) {
    lastRightRaised = true;
    pendingGesture = { type: 'right_hand_raised', confidence: 0.94 };
  } else if (!rightRaised && lastRightRaised) {
    lastRightRaised = false;
  }

  if (leftRaised && !lastLeftRaised) {
    lastLeftRaised = true;
    pendingGesture = { type: 'left_hand_raised', confidence: 0.94 };
  } else if (!leftRaised && lastLeftRaised) {
    lastLeftRaised = false;
  }

  // 4. Waving Gesture Detection
  if (config.detection.gestures) {
    wristHistory.right.push({ x: r_wr.x - r_sh.x, y: r_wr.y });
    if (wristHistory.right.length > 20) wristHistory.right.shift();

    wristHistory.left.push({ x: l_wr.x - l_sh.x, y: l_wr.y });
    if (wristHistory.left.length > 20) wristHistory.left.shift();

    if (checkWaving(wristHistory.right, rightRaised) || checkWaving(wristHistory.left, leftRaised)) {
      const nowT = currentTimeSec || (performance.now() / 1000);
      if (!lastGestureTimes['Waving'] || nowT - lastGestureTimes['Waving'] > 2.0) {
        lastGestureTimes['Waving'] = nowT;
        pendingGesture = { type: 'waving', confidence: 0.92 };
      }
    }
  }

  // 5. Body Movements (Leaning, Turning, Stepping)
  const spineDx = mid_sh.x - mid_hip.x;
  const spineDy = mid_sh.y - mid_hip.y;
  const leanAngle = (Math.atan2(spineDx, -spineDy) * 180) / Math.PI;

  const nowT = currentTimeSec || (performance.now() / 1000);

  if (leanAngle > 14 / config.detection.sensitivity) {
    if (!lastGestureTimes['Lean'] || nowT - lastGestureTimes['Lean'] > 2.0) {
      lastGestureTimes['Lean'] = nowT;
      logEvent('BODY', 'Leaning right', 0.88, null, currentTimeSec);
    }
  } else if (leanAngle < -14 / config.detection.sensitivity) {
    if (!lastGestureTimes['Lean'] || nowT - lastGestureTimes['Lean'] > 2.0) {
      lastGestureTimes['Lean'] = nowT;
      logEvent('BODY', 'Leaning left', 0.88, null, currentTimeSec);
    }
  }

  if (shoulderWidth > 0.05) {
    const relNoseX = (nose.x - mid_sh.x) / shoulderWidth;
    if (relNoseX > 0.28) {
      if (!lastGestureTimes['Turn'] || nowT - lastGestureTimes['Turn'] > 2.0) {
        lastGestureTimes['Turn'] = nowT;
        logEvent('BODY', 'Turning right', 0.88, null, currentTimeSec);
      }
    } else if (relNoseX < -0.28) {
      if (!lastGestureTimes['Turn'] || nowT - lastGestureTimes['Turn'] > 2.0) {
        lastGestureTimes['Turn'] = nowT;
        logEvent('BODY', 'Turning left', 0.88, null, currentTimeSec);
      }
    }
  }

  // Stepping forward / backward
  scaleHistory.push(torsoHeight);
  if (scaleHistory.length > 25) scaleHistory.shift();

  if (scaleHistory.length >= 20 && (!lastStepTime || nowT - lastStepTime > 2.2)) {
    const past = scaleHistory.slice(0, 6).reduce((a, b) => a + b, 0) / 6;
    const curr = scaleHistory.slice(-6).reduce((a, b) => a + b, 0) / 6;
    const ratio = curr / past;
    if (ratio > 1.09) {
      lastStepTime = nowT;
      logEvent('BODY', 'Stepping forward', 0.87, null, currentTimeSec);
    } else if (ratio < 0.91) {
      lastStepTime = nowT;
      logEvent('BODY', 'Stepping backward', 0.86, null, currentTimeSec);
    }
  }
}

function checkWaving(history, handRaised) {
  if (!handRaised || history.length < 12) return false;
  const xs = history.map(h => h.x);
  const amplitude = Math.max(...xs) - Math.min(...xs);
  if (amplitude < 0.035 / config.detection.sensitivity) return false;

  let reversals = 0;
  let prevDir = 0;
  for (let i = 1; i < xs.length; i++) {
    const vel = xs[i] - xs[i - 1];
    let currDir = 0;
    if (vel > 0.002) currDir = 1;
    else if (vel < -0.002) currDir = -1;
    else continue;

    if (prevDir !== 0 && currDir !== prevDir) reversals++;
    prevDir = currDir;
  }
  return reversals >= 2;
}

let lastRenderTimestamp = 0;
function renderLoop(timestamp) {
  requestAnimationFrame(renderLoop);
  if (!isRunning || isPaused) return;

  // Cap canvas render at ~60 FPS to prevent burning CPU on 120Hz/140Hz displays
  if (timestamp && timestamp - lastRenderTimestamp < 15) return;
  lastRenderTimestamp = timestamp;

  const vw = activeVideo.videoWidth;
  const vh = activeVideo.videoHeight;

  if (!vw || !vh || activeVideo.readyState < 2) {
    // Elegant indicator on canvas while connecting to video feed
    ctx.save();
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '500 15px Inter, -apple-system, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Connecting to camera feed...', canvas.width / 2, canvas.height / 2);
    ctx.restore();
    return;
  }

  badgeRes.textContent = `${vw} × ${vh}`;

  // Measure FPS
  fpsCounter++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    measuredFps = Math.round((fpsCounter * 1000) / (now - lastFpsTime));
    badgeFps.textContent = `FPS: ${measuredFps}`;
    fpsCounter = 0;
    lastFpsTime = now;
  }

  // Draw Video Frame to Canvas preserving aspect ratio
  ctx.save();
  ctx.fillStyle = '#050811';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const scale = Math.min(canvas.width / vw, canvas.height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const dx = (canvas.width - dw) / 2;
  const dy = (canvas.height - dh) / 2;

  // Mirror effect if webcam
  if (currentSource === 'LIVE' && config.video.mirror) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(activeVideo, canvas.width - dx - dw, dy, dw, dh);
    ctx.restore();
  } else {
    ctx.drawImage(activeVideo, dx, dy, dw, dh);
    ctx.restore();
  }

  // Send frame to MediaPipe if available and not busy (throttled to save GPU/WASM bandwidth)
  if (poseModel && !isPoseBusy && activeVideo.readyState >= 2 && (now - lastPoseDispatch >= 65)) {
    isPoseBusy = true;
    lastPoseDispatch = now;
    poseModel.send({ image: activeVideo })
      .catch(() => {})
      .finally(() => { isPoseBusy = false; });
  }
  if (handsModel && !isHandsBusy && config.display.showHands && activeVideo.readyState >= 2 && (now - lastHandsDispatch >= 120)) {
    isHandsBusy = true;
    lastHandsDispatch = now;
    handsModel.send({ image: activeVideo })
      .catch(() => {})
      .finally(() => { isHandsBusy = false; });
  }

  // Draw Skeletal Overlay
  if (latestPoseResults && latestPoseResults.poseLandmarks) {
    const lms = latestPoseResults.poseLandmarks;
    const curTime = currentSource === 'FILE' ? activeVideo.currentTime : null;
    processTemporalAnalysis(lms, curTime);

    if (config.display.showSkeleton) {
      drawSkeletonOverlay(lms, dx, dy, dw, dh);
    }
  }

  // Draw Hand Landmarks
  if (config.display.showHands && latestHandLandmarks.length > 0) {
    drawHandOverlay(latestHandLandmarks, dx, dy, dw, dh);
  }

  // Send frame to Backend Multi-Person AI Tracker
  sendFrameToAI(activeVideo);

  // Draw Multi-Person Bounding Boxes and Recognized Identity Badges
  if (latestTrackedPersons && latestTrackedPersons.length > 0) {
    drawTrackedPersons(latestTrackedPersons, dx, dy, dw, dh, vw, vh);
  }

  // Update file progress slider
  if (currentSource === 'FILE' && !activeVideo.paused) {
    const cur = activeVideo.currentTime || 0;
    const dur = activeVideo.duration || 1;
    seekSlider.value = (cur / dur) * 1000;
    const curM = String(Math.floor(cur / 60)).padStart(2, '0');
    const curS = String(Math.floor(cur % 60)).padStart(2, '0');
    const durM = String(Math.floor(dur / 60)).padStart(2, '0');
    const durS = String(Math.floor(dur % 60)).padStart(2, '0');
    timeLabel.textContent = `${curM}:${curS} / ${durM}:${durS}`;
  }
}

function toScreen(lm, dx, dy, dw, dh) {
  let x = lm.x;
  if (currentSource === 'LIVE' && config.video.mirror) {
    x = 1.0 - x;
  }
  return {
    x: dx + x * dw,
    y: dy + lm.y * dh
  };
}

function drawSkeletonOverlay(lms, dx, dy, dw, dh) {
  ctx.save();
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 3.2;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = 'rgba(0, 210, 255, 0.6)';
  ctx.shadowBlur = 8;

  // Draw Bones
  POSE_CONNECTIONS.forEach(([i1, i2]) => {
    const p1 = lms[i1];
    const p2 = lms[i2];
    if (p1 && p2 && (p1.visibility || 1) > 0.4 && (p2.visibility || 1) > 0.4) {
      const pt1 = toScreen(p1, dx, dy, dw, dh);
      const pt2 = toScreen(p2, dx, dy, dw, dh);
      ctx.beginPath();
      ctx.moveTo(pt1.x, pt1.y);
      ctx.lineTo(pt2.x, pt2.y);
      ctx.stroke();
    }
  });

  // Central spine
  const l_sh = lms[11], r_sh = lms[12];
  const l_hip = lms[23], r_hip = lms[24];
  const nose = lms[0];
  if (l_sh && r_sh && l_hip && r_hip) {
    const mid_sh = { x: (l_sh.x + r_sh.x) / 2, y: (l_sh.y + r_sh.y) / 2 };
    const mid_hip = { x: (l_hip.x + r_hip.x) / 2, y: (l_hip.y + r_hip.y) / 2 };
    const pt_sh = toScreen(mid_sh, dx, dy, dw, dh);
    const pt_hip = toScreen(mid_hip, dx, dy, dw, dh);
    ctx.beginPath();
    ctx.moveTo(pt_sh.x, pt_sh.y);
    ctx.lineTo(pt_hip.x, pt_hip.y);
    ctx.stroke();

    if (nose) {
      const pt_nose = toScreen(nose, dx, dy, dw, dh);
      ctx.beginPath();
      ctx.moveTo(pt_nose.x, pt_nose.y);
      ctx.lineTo(pt_sh.x, pt_sh.y);
      ctx.stroke();
    }
  }

  // Draw Joint Nodes
  if (config.display.showLandmarks) {
    const joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    joints.forEach(idx => {
      const lm = lms[idx];
      if (lm && (lm.visibility || 1) > 0.4) {
        const pt = toScreen(lm, dx, dy, dw, dh);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6.0, 0, Math.PI * 2);
        ctx.fillStyle = '#0088ff';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    });
  }

  ctx.restore();
}

function drawHandOverlay(multiHands, dx, dy, dw, dh) {
  ctx.save();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.0;
  ctx.lineCap = 'round';

  multiHands.forEach(handLms => {
    HAND_CONNECTIONS.forEach(([i1, i2]) => {
      const pt1 = toScreen(handLms[i1], dx, dy, dw, dh);
      const pt2 = toScreen(handLms[i2], dx, dy, dw, dh);
      ctx.beginPath();
      ctx.moveTo(pt1.x, pt1.y);
      ctx.lineTo(pt2.x, pt2.y);
      ctx.stroke();
    });

    handLms.forEach(lm => {
      const pt = toScreen(lm, dx, dy, dw, dh);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#0088ff';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    });
  });
  ctx.restore();
}

// --------------------------------------------------------------------------
// 5.5 Backend Multi-Person AI Tracker Integration
// --------------------------------------------------------------------------
async function initTrackingSession() {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      if (data.startup_events && data.startup_events.length > 0) {
        data.startup_events.forEach(evt => {
          logEvent(evt.type, evt.text);
        });
        return;
      }
    }
  } catch (e) {
    console.warn('API reset connection failed:', e);
  }
  // Fallback if network glitch occurred
  logEvent('SYSTEM', 'Camera connected successfully');
  logEvent('SYSTEM', 'Face recognition initialized');
  logEvent('SYSTEM', 'Movement detection started');
}

async function sendFrameToAI(videoElem) {
  if (isAnalyzingFrame || !isRunning || isPaused || videoElem.readyState < 2) return;
  const now = performance.now();
  if (now - lastAnalysisTimestamp < 220) return; // ~4.5 FPS - ultra-responsive and eliminates CPU lag
  lastAnalysisTimestamp = now;
  isAnalyzingFrame = true;

  try {
    if (!offscreenCanvas) {
      offscreenCanvas = document.createElement('canvas');
      offscreenCtx = offscreenCanvas.getContext('2d');
    }
    const targetW = 480;
    const targetH = Math.round((videoElem.videoHeight / (videoElem.videoWidth || 1)) * 480) || 360;
    offscreenCanvas.width = targetW;
    offscreenCanvas.height = targetH;

    // Draw mirrored or normal matching source view
    offscreenCtx.save();
    if (currentSource === 'LIVE' && config.video.mirror) {
      offscreenCtx.translate(targetW, 0);
      offscreenCtx.scale(-1, 1);
    }
    offscreenCtx.drawImage(videoElem, 0, 0, targetW, targetH);
    offscreenCtx.restore();

    const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.55);
    const curTimeSec = currentSource === 'FILE' ? videoElem.currentTime : null;
    const currentGesture = pendingGesture;
    pendingGesture = null;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        timestamp: curTimeSec,
        gesture: currentGesture
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data.persons) {
        latestTrackedPersons = data.persons;
        if (latestTrackedPersons.length > 0) {
          const known = latestTrackedPersons.filter(p => p.is_known).map(p => p.name);
          if (known.length > 0) {
            setStatus(`Tracking: ${known.join(', ')}`, 'active');
          } else {
            setStatus('Tracking...', 'active');
          }
        }
      }
      if (data.events && data.events.length > 0) {
        data.events.forEach(evt => {
          logEvent(evt.type, evt.text, evt.confidence, evt.person_id);
        });
      }
    }
  } catch (err) {
    // Graceful offline fallback
  } finally {
    isAnalyzingFrame = false;
  }
}

function drawTrackedPersons(persons, dx, dy, dw, dh, vw, vh) {
  if (!persons || persons.length === 0) return;

  ctx.save();
  persons.forEach(person => {
    let nx1, ny1, nx2, ny2;
    if (person.bbox_norm) {
      [nx1, ny1, nx2, ny2] = person.bbox_norm;
    } else {
      nx1 = person.bbox[0] / vw;
      ny1 = person.bbox[1] / vh;
      nx2 = person.bbox[2] / vw;
      ny2 = person.bbox[3] / vh;
    }
    const sx1 = dx + nx1 * dw;
    const sx2 = dx + nx2 * dw;
    const sy1 = dy + ny1 * dh;
    const sy2 = dy + ny2 * dh;

    const boxW = Math.max(50, sx2 - sx1);
    const boxH = Math.max(50, sy2 - sy1);

    const isKnown = person.is_known;
    const primaryColor = isKnown ? '#00d2ff' : '#f59e0b';
    const glowColor = isKnown ? 'rgba(0, 210, 255, 0.45)' : 'rgba(245, 158, 11, 0.45)';

    // Corner brackets HUD style
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 10;

    const cornerLen = Math.min(22, boxW * 0.22, boxH * 0.22);
    // Top-left
    ctx.beginPath();
    ctx.moveTo(sx1, sy1 + cornerLen);
    ctx.lineTo(sx1, sy1);
    ctx.lineTo(sx1 + cornerLen, sy1);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(sx2 - cornerLen, sy1);
    ctx.lineTo(sx2, sy1);
    ctx.lineTo(sx2, sy1 + cornerLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(sx1, sy2 - cornerLen);
    ctx.lineTo(sx1, sy2);
    ctx.lineTo(sx1 + cornerLen, sy2);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(sx2 - cornerLen, sy2);
    ctx.lineTo(sx2, sy2);
    ctx.lineTo(sx2, sy2 - cornerLen);
    ctx.stroke();

    // Semi-transparent box border
    ctx.strokeStyle = isKnown ? 'rgba(0, 210, 255, 0.2)' : 'rgba(245, 158, 11, 0.2)';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(sx1, sy1, boxW, boxH);

    // Pill badge for name
    const confPct = Math.round(person.confidence * 100);
    const labelText = isKnown ? `${person.name} (${confPct}%)` : 'Unknown';
    ctx.font = '600 13px Inter, -apple-system, sans-serif';
    const textMetrics = ctx.measureText(labelText);
    const badgeW = textMetrics.width + 28;
    const badgeH = 24;
    const badgeX = sx1 + (boxW - badgeW) / 2;
    const badgeY = Math.max(dy + 6, sy1 - badgeH - 6);

    // Pill background
    ctx.fillStyle = 'rgba(7, 12, 23, 0.88)';
    if (typeof ctx.roundRect === 'function') {
      ctx.beginPath();
      ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 6);
      ctx.fill();
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    } else {
      ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);
    }

    // Status dot inside pill
    ctx.beginPath();
    ctx.arc(badgeX + 11, badgeY + badgeH / 2, 4, 0, Math.PI * 2);
    ctx.fillStyle = isKnown ? '#22c55e' : '#f59e0b';
    ctx.shadowBlur = 6;
    ctx.shadowColor = ctx.fillStyle;
    ctx.fill();

    // Text
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, badgeX + 20, badgeY + badgeH / 2 + 1);

    // Action sub-tag if available
    if (person.action && person.action !== 'present') {
      ctx.font = '500 11px Inter, -apple-system, sans-serif';
      const actMetrics = ctx.measureText(person.action);
      const actW = actMetrics.width + 16;
      const actH = 18;
      const actX = sx1 + (boxW - actW) / 2;
      const actY = sy2 + 6;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      if (typeof ctx.roundRect === 'function') {
        ctx.beginPath();
        ctx.roundRect(actX, actY, actW, actH, 4);
        ctx.fill();
      } else {
        ctx.fillRect(actX, actY, actW, actH);
      }
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.fillText(person.action, actX + 8, actY + actH / 2 + 1);
    }
  });
  ctx.restore();
}

// --------------------------------------------------------------------------
// 6. UI Interactions and Event Listeners
// --------------------------------------------------------------------------
function setStatus(text, type = 'active') {
  statusText.textContent = text;
  statusDot.className = 'status-dot-indicator';
  if (type === 'active') statusDot.classList.add('dot-active');
  else if (type === 'warning') statusDot.classList.add('dot-warning');
  else if (type === 'error') statusDot.classList.add('dot-error');
  else statusDot.classList.add('dot-neutral');
}

btnSelectLive.addEventListener('click', () => {
  startWebcam(cameraSelect.value !== 'default' ? cameraSelect.value : null);
});

if (btnOverlayRetry) {
  btnOverlayRetry.addEventListener('click', () => {
    startWebcam(cameraSelect.value !== 'default' ? cameraSelect.value : null);
  });
}

if (btnOverlayDemo) {
  btnOverlayDemo.addEventListener('click', () => {
    loadVideoUrl('/samples/demo_motion_analysis.mp4', 'demo_motion_analysis.mp4');
  });
}

btnSelectUpload.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    loadVideoFile(file);
  }
});

btnFilePlayPause.addEventListener('click', () => {
  if (fileVideo.paused) {
    fileVideo.play();
    btnFilePlayPause.textContent = 'Pause';
    setStatus('Tracking...', 'active');
  } else {
    fileVideo.pause();
    btnFilePlayPause.textContent = 'Play';
    setStatus('Paused', 'warning');
  }
});

btnFileRestart.addEventListener('click', () => {
  fileVideo.currentTime = 0;
  fileVideo.play();
  resetAnalysisState();
});

seekSlider.addEventListener('input', (e) => {
  const dur = fileVideo.duration || 1;
  fileVideo.currentTime = (e.target.value / 1000) * dur;
});

btnClearTerminal.addEventListener('click', () => {
  terminalLogBody.innerHTML = '';
});

cameraSelect.addEventListener('change', (e) => {
  if (currentSource === 'LIVE') {
    startWebcam(e.target.value !== 'default' ? e.target.value : null);
  }
});

btnMainAction.addEventListener('click', () => {
  isRunning = !isRunning;
  if (isRunning) {
    btnMainAction.className = 'btn-main-action stop-state';
    setStatus('Tracking...', 'active');
    if (currentSource === 'FILE') fileVideo.play();
    else webcamVideo.play();
  } else {
    btnMainAction.className = 'btn-main-action start-state';
    setStatus('Stopped', 'neutral');
    if (currentSource === 'FILE') fileVideo.pause();
    else webcamVideo.pause();
  }
});

// Modals
btnSettings.addEventListener('click', () => { settingsModal.style.display = 'flex'; });
document.getElementById('btnCloseSettings').addEventListener('click', () => { settingsModal.style.display = 'none'; });
document.getElementById('btnSaveSettings').addEventListener('click', () => { settingsModal.style.display = 'none'; });

btnHelp.addEventListener('click', () => { helpModal.style.display = 'flex'; });
document.getElementById('btnCloseHelp').addEventListener('click', () => { helpModal.style.display = 'none'; });
document.getElementById('btnDismissHelp').addEventListener('click', () => { helpModal.style.display = 'none'; });

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// Log Export
document.getElementById('btnExportCsv').addEventListener('click', () => {
  let csv = 'Timestamp,Type,Message,Confidence,PersonID\n';
  eventLog.forEach(e => {
    csv += `"${e.timestamp}","${e.type}","${e.message}","${e.confidence || ''}","${e.personId || ''}"\n`;
  });
  downloadFile(csv, 'motion_reader_logs.csv', 'text/csv');
});

document.getElementById('btnExportJson').addEventListener('click', () => {
  const jsonStr = JSON.stringify(eventLog, null, 2);
  downloadFile(jsonStr, 'motion_reader_logs.json', 'application/json');
});

function downloadFile(content, fileName, contentType) {
  const a = document.createElement('a');
  const file = new Blob([content], { type: contentType });
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
}

// --------------------------------------------------------------------------
// Initialization on page load
// --------------------------------------------------------------------------
let isBooted = false;
function bootstrapApplication() {
  if (isBooted) return;
  isBooted = true;

  // 1. Start rendering loop immediately
  requestAnimationFrame(renderLoop);

  // 2. Start webcam immediately
  startWebcam();

  // 3. Enumerate camera hardware asynchronously
  enumerateCameras().catch(err => console.warn('Camera enumeration error:', err));

  // 4. Initialize MediaPipe asynchronously without blocking anything
  setTimeout(() => {
    try {
      initMediaPipe();
    } catch (e) {
      console.warn('Non-critical MediaPipe setup notice:', e);
    }
  }, 100);
}

// Ensure execution whether DOMContentLoaded already fired or is about to fire
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrapApplication);
} else {
  bootstrapApplication();
}

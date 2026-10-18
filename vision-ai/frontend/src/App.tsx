import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Header } from './components/Header/Header';
import { InputSwitcher } from './components/InputSwitcher/InputSwitcher';
import { VideoPanel } from './components/VideoPanel/VideoPanel';
import { AnalysisTerminal } from './components/Terminal/AnalysisTerminal';
import { StatusPanel } from './components/StatusPanel/StatusPanel';
import { SettingsPanel } from './components/Settings/SettingsPanel';
import { useAnalysisWS } from './hooks/useAnalysisWS';
import { api } from './services/api';
import type {
  InputMode, SessionStatus, AnalysisEvent, FrameStats, VideoPosition,
  AppSettings, UploadedFile
} from './types';
import { DEFAULT_SETTINGS } from './types';

// ─── Camera utilities ─────────────────────────────────────────────────────────

async function enumerateCameras(): Promise<{ index: number; label: string }[]> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((d) => d.kind === 'videoinput');
    return cameras.map((d, i) => ({
      index: i,
      label: d.label || `Camera ${i + 1}`,
    }));
  } catch {
    return [{ index: 0, label: 'Default Camera' }];
  }
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);

  const [inputMode, setInputMode] = useState<InputMode>('live_camera');
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>('idle');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);

  const [events, setEvents] = useState<AnalysisEvent[]>([]);
  const [frameStats, setFrameStats] = useState<FrameStats>({
    type: 'frame_stats', session_id: '', fps: 0, latency_ms: 0,
    person_count: 0, face_count: 0, hand_count: 0,
    processing_mode: 'balanced', device: 'cpu', timestamp_ms: 0,
  });
  const [videoPos, setVideoPos] = useState<VideoPosition | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState<Record<string, boolean>>({
    pose: false, face: false, hands: false
  });
  const [currentPosture, setCurrentPosture] = useState('');

  // Camera / video state
  const [cameras, setCameras] = useState<{ index: number; label: string }[]>([]);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | undefined>();
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [autoScroll, setAutoScroll] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);

  const streamRef = useRef<HTMLVideoElement>(null);
  const streamMediaRef = useRef<MediaStream | null>(null);

  // ── Backend health check ────────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        await api.health();
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Camera enumeration ─────────────────────────────────────────────────────
  useEffect(() => {
    enumerateCameras().then(setCameras);
  }, []);

  // ── WebSocket ──────────────────────────────────────────────────────────────
  const { wsStatus, sendStart, sendStop, sendPause, sendResume, sendSeek } = useAnalysisWS({
    sessionId,
    onEvent: useCallback((ev: AnalysisEvent) => {
      setEvents((prev) => {
        const next = [...prev, ev];
        // Extract posture from events
        if (ev.category === 'POSTURE') {
          const match = ev.action.match(/posture_(.+)/);
          if (match) setCurrentPosture(match[1].charAt(0).toUpperCase() + match[1].slice(1));
        }
        return next.slice(-settings.terminalMaxLines);
      });
    }, [settings.terminalMaxLines]),
    onStats: useCallback((s: FrameStats) => setFrameStats(s), []),
    onVideoPosition: useCallback((pos: VideoPosition) => setVideoPos(pos), []),
    onStatusChange: useCallback((status: string, _message: string, meta?: Record<string, unknown>) => {
      if (status === 'running') setSessionStatus('running');
      else if (status === 'paused') setSessionStatus('paused');
      else if (status === 'stopped') setSessionStatus('stopped');
      else if (status === 'connected') {/* noop */}
      if (meta?.models_loaded) {
        setModelsLoaded(meta.models_loaded as Record<string, boolean>);
      }
    }, []),
  });

  // ── Camera stream ──────────────────────────────────────────────────────────
  const startCameraStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: selectedCamera > 0 ? { exact: String(selectedCamera) } : undefined }
      });
      streamMediaRef.current = stream;
      if (streamRef.current) {
        streamRef.current.srcObject = stream;
        streamRef.current.play().catch(() => {});
      }
      // Refresh camera list now that we have permission
      enumerateCameras().then(setCameras);
    } catch (err) {
      console.error('[Camera] Access denied:', err);
      setEvents((prev) => [...prev, {
        type: 'movement_event' as const,
        session_id: sessionId || '',
        timestamp_ms: 0,
        category: 'ERROR' as const,
        action: 'camera_permission_denied',
        description: 'Camera access denied. Please grant camera permission in your browser.',
        source: 'browser',
        severity: 'alert' as const,
      }]);
    }
  }, [selectedCamera, sessionId]);

  const stopCameraStream = useCallback(() => {
    if (streamMediaRef.current) {
      streamMediaRef.current.getTracks().forEach((t) => t.stop());
      streamMediaRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.srcObject = null;
    }
  }, []);

  // ── Video file upload ──────────────────────────────────────────────────────
  const handleFileSelect = useCallback(async (file: File) => {
    if (uploading) return;
    setUploading(true);
    try {
      const result = await api.uploadVideo(file);
      setUploadedFile({
        serverPath: `uploads/${result.session_id}`,
        filename: result.filename,
        size_bytes: result.size_bytes,
        duration_seconds: result.duration_seconds ?? undefined,
      });
      const url = URL.createObjectURL(file);
      if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
      setVideoObjectUrl(url);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setEvents((prev) => [...prev, {
        type: 'movement_event' as const,
        session_id: sessionId || '',
        timestamp_ms: 0,
        category: 'ERROR' as const,
        action: 'upload_error',
        description: `Upload failed: ${msg}`,
        source: 'system',
        severity: 'alert' as const,
      }]);
    } finally {
      setUploading(false);
    }
  }, [uploading, videoObjectUrl, sessionId]);

  // ── Mode switch ────────────────────────────────────────────────────────────
  const handleModeChange = useCallback(async (mode: InputMode) => {
    if (sessionStatus === 'running' || sessionStatus === 'paused') {
      await handleStop();
    }
    setInputMode(mode);
    setUploadedFile(null);
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    setVideoObjectUrl(undefined);
    setEvents([]);
    setCurrentPosture('');
    setVideoPos(null);
    if (mode === 'live_camera') stopCameraStream();
  }, [sessionStatus, videoObjectUrl, stopCameraStream]);

  // ── Analysis controls ──────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (!backendOnline) {
      alert('Backend server is offline. Start it with: cd vision-ai/backend && python -m uvicorn app.main:app --port 8080');
      return;
    }

    setSessionStatus('initializing');
    setEvents([]);
    setCurrentPosture('');

    try {
      const config = {
        input_type: inputMode,
        source_name: inputMode === 'live_camera' ? `Camera ${selectedCamera}` : (uploadedFile?.filename || ''),
        processing_mode: settings.processingMode,
        enable_face: settings.enableFace,
        enable_pose: settings.enablePose,
        enable_hands: settings.enableHands,
        enable_ai_descriptions: settings.enableAI,
        camera_index: selectedCamera,
        inference_width: 640,
        inference_height: 480,
        ...(inputMode === 'uploaded_video' && uploadedFile
          ? { video_path: `uploads/${uploadedFile.serverPath.replace(/^uploads\//, '')}` }
          : {}),
      };

      const { session_id } = await api.startSession(config);
      setSessionId(session_id);

      if (inputMode === 'live_camera') {
        await startCameraStream();
      }

      // WS connects when sessionId updates; sendStart is queued and transmitted on connection
      sendStart(config);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSessionStatus('error');
      setEvents((prev) => [...prev, {
        type: 'movement_event' as const,
        session_id: '',
        timestamp_ms: 0,
        category: 'ERROR' as const,
        action: 'session_error',
        description: `Failed to start session: ${msg}`,
        source: 'system',
        severity: 'alert' as const,
      }]);
    }
  }, [
    backendOnline, inputMode, selectedCamera, uploadedFile, settings,
    startCameraStream, sendStart,
  ]);

  const handlePause = useCallback(() => {
    if (sessionStatus === 'paused') {
      sendResume();
      setSessionStatus('running');
    } else {
      sendPause();
      setSessionStatus('paused');
    }
  }, [sessionStatus, sendPause, sendResume]);

  const handleStop = useCallback(async () => {
    sendStop();
    if (sessionId) await api.stopSession(sessionId).catch(() => {});
    stopCameraStream();
    setSessionStatus('stopped');
    setSessionId(null);
  }, [sendStop, sessionId, stopCameraStream]);

  const handleReset = useCallback(async () => {
    await handleStop();
    setEvents([]);
    setCurrentPosture('');
    setVideoPos(null);
    setSessionStatus('idle');
  }, [handleStop]);

  const handleSeek = useCallback((progress: number) => {
    sendSeek(progress);
  }, [sendSeek]);

  // ── Drag-and-drop ──────────────────────────────────────────────────────────
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  // ── Cleanup ────────────────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopCameraStream();
      if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    };
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-bg-primary">
      {/* Header */}
      <Header
        sessionStatus={sessionStatus}
        wsStatus={wsStatus}
        backendOnline={backendOnline}
        onSettingsClick={() => setShowSettings(true)}
      />

      {/* Input switcher toolbar */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-border-subtle bg-bg-secondary shrink-0">
        <InputSwitcher
          mode={inputMode}
          onModeChange={handleModeChange}
          cameras={cameras}
          selectedCamera={selectedCamera}
          onCameraChange={setSelectedCamera}
          uploadedFile={uploadedFile}
          onFileSelect={handleFileSelect}
          isDragging={isDragging}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          uploading={uploading}
        />
      </div>

      {/* Main content */}
      <div className="flex flex-1 min-h-0 gap-0">
        {/* Left: Video panel */}
        <div className="flex flex-col w-[55%] min-w-0 border-r border-border-subtle">
          <div className="flex-1 p-3 min-h-0">
            <VideoPanel
              inputMode={inputMode}
              sessionStatus={sessionStatus}
              videoSrc={videoObjectUrl}
              streamRef={streamRef}
              personCount={frameStats.person_count}
              faceCount={frameStats.face_count}
              handCount={frameStats.hand_count}
              fps={frameStats.fps}
              latencyMs={frameStats.latency_ms}
              showOverlay={showOverlay}
              onToggleOverlay={() => setShowOverlay((v) => !v)}
              onStart={handleStart}
              onPause={handlePause}
              onStop={handleStop}
              onReset={handleReset}
              videoProgress={videoPos?.progress}
              videoDuration={videoPos?.duration_s}
              videoCurrentS={videoPos?.current_s}
              onSeek={handleSeek}
            />
          </div>
        </div>

        {/* Right: Terminal + Status */}
        <div className="flex flex-col flex-1 min-w-0 min-h-0">
          {/* Terminal (top 65%) */}
          <div className="flex-[65] p-3 min-h-0 border-b border-border-subtle">
            <AnalysisTerminal
              events={events}
              autoScroll={autoScroll}
              onToggleAutoScroll={() => setAutoScroll((v) => !v)}
              onClear={() => setEvents([])}
              sessionId={sessionId}
            />
          </div>

          {/* Status panel (bottom 35%) */}
          <div className="flex-[35] p-3 min-h-0">
            <StatusPanel
              personCount={frameStats.person_count}
              faceCount={frameStats.face_count}
              handCount={frameStats.hand_count}
              fps={frameStats.fps}
              latencyMs={frameStats.latency_ms}
              processingMode={frameStats.processing_mode}
              device={frameStats.device}
              modelsLoaded={modelsLoaded}
              currentPosture={currentPosture}
              eventCount={events.length}
            />
          </div>
        </div>
      </div>

      {/* Settings modal */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onUpdate={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}

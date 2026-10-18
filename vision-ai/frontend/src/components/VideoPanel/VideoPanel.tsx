import React, { useRef, useEffect, useCallback } from 'react';
import { Play, Pause, Square, RotateCcw, Eye, EyeOff, Camera, AlertCircle, Loader2 } from 'lucide-react';
import type { SessionStatus, InputMode } from '../../types';

interface VideoPanelProps {
  inputMode: InputMode;
  sessionStatus: SessionStatus;
  videoSrc?: string;         // object URL for uploaded video
  streamRef?: React.RefObject<HTMLVideoElement | null>;
  personCount: number;
  faceCount: number;
  handCount: number;
  fps: number;
  latencyMs: number;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onReset: () => void;
  videoProgress?: number;
  videoDuration?: number;
  videoCurrentS?: number;
  onSeek?: (progress: number) => void;
}

export const VideoPanel: React.FC<VideoPanelProps> = ({
  inputMode, sessionStatus, videoSrc, streamRef,
  personCount, faceCount, handCount, fps, latencyMs,
  showOverlay, onToggleOverlay,
  onStart, onPause, onStop, onReset,
  videoProgress = 0, videoDuration = 0, videoCurrentS = 0, onSeek,
}) => {
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const isRunning = sessionStatus === 'running';
  const isPaused = sessionStatus === 'paused';
  const isActive = isRunning || isPaused;

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${String(sec).padStart(2, '0')}`;
  };

  // Sync videoSrc to local video element
  useEffect(() => {
    if (localVideoRef.current && videoSrc) {
      localVideoRef.current.src = videoSrc;
      localVideoRef.current.load();
    }
  }, [videoSrc]);

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!onSeek || videoDuration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const progress = (e.clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(1, progress)));
  }, [onSeek, videoDuration]);

  return (
    <div className="panel flex flex-col overflow-hidden h-full">
      {/* Panel header */}
      <div className="panel-header">
        <span className="panel-label">
          <div className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-green-accent animate-pulse' : isPaused ? 'bg-amber-accent' : 'bg-text-dim'}`} />
          {inputMode === 'live_camera' ? 'Camera Feed' : 'Video Analysis'}
        </span>

        <div className="flex items-center gap-1">
          {/* Overlay toggle */}
          <button
            onClick={onToggleOverlay}
            className="btn-ghost px-2 py-1 gap-1 text-[11px]"
            title="Toggle detection overlays"
          >
            {showOverlay ? <Eye size={12} /> : <EyeOff size={12} />}
            <span className="hidden sm:inline">Overlays</span>
          </button>
        </div>
      </div>

      {/* Video area */}
      <div className="relative flex-1 bg-black overflow-hidden min-h-0">
        {/* Camera stream */}
        {inputMode === 'live_camera' && (
          <video
            ref={streamRef || localVideoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-contain transition-opacity duration-300 ${isActive ? 'opacity-100' : 'opacity-40'}`}
          />
        )}

        {/* Video file */}
        {inputMode === 'uploaded_video' && videoSrc && (
          <video
            ref={localVideoRef}
            className="w-full h-full object-contain"
            playsInline
          />
        )}

        {/* Idle / No signal state */}
        {!isActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            {sessionStatus === 'initializing' ? (
              <>
                <Loader2 size={32} className="text-cyan-accent animate-spin" />
                <p className="text-xs text-text-muted">Initializing AI models...</p>
              </>
            ) : sessionStatus === 'error' ? (
              <>
                <AlertCircle size={32} className="text-red-accent" />
                <p className="text-xs text-red-400">Analysis error — check terminal</p>
              </>
            ) : (
              <>
                <div className="w-16 h-16 rounded-full border-2 border-dashed border-text-dim flex items-center justify-center">
                  <Camera size={24} className="text-text-dim" />
                </div>
                <p className="text-xs text-text-dim">
                  {inputMode === 'live_camera' ? 'Start analysis to activate camera' : 'Load a video file and start analysis'}
                </p>
              </>
            )}
          </div>
        )}

        {/* HUD Overlays */}
        {isActive && showOverlay && (
          <>
            {/* Top-left: Person count */}
            <div className="absolute top-2 left-2 flex flex-col gap-1">
              <div className="flex items-center gap-1.5 bg-black/60 rounded px-2 py-1 backdrop-blur-sm">
                <div className="w-1.5 h-1.5 rounded-full bg-green-accent animate-pulse" />
                <span className="text-[11px] font-mono text-green-400">
                  {personCount} {personCount === 1 ? 'Person' : 'Persons'}
                </span>
              </div>
              {faceCount > 0 && (
                <div className="bg-black/60 rounded px-2 py-0.5 backdrop-blur-sm">
                  <span className="text-[10px] font-mono text-cyan-400">{faceCount} Face{faceCount > 1 ? 's' : ''}</span>
                </div>
              )}
              {handCount > 0 && (
                <div className="bg-black/60 rounded px-2 py-0.5 backdrop-blur-sm">
                  <span className="text-[10px] font-mono text-pink-400">{handCount} Hand{handCount > 1 ? 's' : ''}</span>
                </div>
              )}
            </div>

            {/* Top-right: Performance */}
            <div className="absolute top-2 right-2 flex flex-col items-end gap-1">
              <div className="bg-black/60 rounded px-2 py-1 backdrop-blur-sm">
                <span className="text-[11px] font-mono text-text-secondary">
                  {fps.toFixed(0)} <span className="text-text-dim">fps</span>
                </span>
              </div>
              <div className="bg-black/60 rounded px-2 py-0.5 backdrop-blur-sm">
                <span className="text-[10px] font-mono text-text-dim">{latencyMs.toFixed(0)}ms</span>
              </div>
            </div>

            {/* Live indicator */}
            {inputMode === 'live_camera' && (
              <div className="absolute top-2 left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-red-accent/80 rounded px-2 py-0.5">
                <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                <span className="text-[10px] font-semibold text-white tracking-widest">LIVE</span>
              </div>
            )}
          </>
        )}
      </div>

      {/* Video progress bar (uploaded video only) */}
      {inputMode === 'uploaded_video' && videoDuration > 0 && (
        <div className="px-3 py-2 border-t border-border-subtle">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-text-muted w-10">{formatTime(videoCurrentS)}</span>
            <div
              className="flex-1 h-1 bg-bg-card rounded cursor-pointer relative"
              onClick={handleSeek}
            >
              <div
                className="h-full bg-cyan-accent rounded transition-all"
                style={{ width: `${(videoProgress * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-text-muted w-10 text-right">{formatTime(videoDuration)}</span>
          </div>
        </div>
      )}

      {/* Controls bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-t border-border-subtle">
        {!isActive ? (
          <button onClick={onStart} className="btn-primary">
            <Play size={12} />
            Start Analysis
          </button>
        ) : (
          <>
            <button onClick={onPause} className="btn-secondary">
              {isPaused ? <><Play size={12} /> Resume</> : <><Pause size={12} /> Pause</>}
            </button>
            <button onClick={onStop} className="btn-danger">
              <Square size={12} />
              Stop
            </button>
          </>
        )}
        <button onClick={onReset} className="btn-ghost ml-auto">
          <RotateCcw size={12} />
          Reset
        </button>
      </div>
    </div>
  );
};

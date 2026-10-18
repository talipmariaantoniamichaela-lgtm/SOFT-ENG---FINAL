// VISION AI — Shared TypeScript types

export type InputMode = 'live_camera' | 'uploaded_video';
export type ProcessingMode = 'performance' | 'balanced' | 'accuracy';
export type SessionStatus = 'idle' | 'initializing' | 'running' | 'paused' | 'stopped' | 'error';

export interface SessionConfig {
  input_type: InputMode;
  source_name: string;
  processing_mode: ProcessingMode;
  enable_face: boolean;
  enable_pose: boolean;
  enable_hands: boolean;
  enable_ai_descriptions: boolean;
  camera_index: number;
  inference_width: number;
  inference_height: number;
  video_path?: string;
}

export interface AnalysisSession {
  session_id: string;
  input_type: InputMode;
  source_name: string;
  status: SessionStatus;
  event_count: number;
  started_at: number;
  config: SessionConfig;
}

export type EventCategory =
  | 'SYSTEM' | 'DETECT' | 'FACE' | 'ACTION'
  | 'POSTURE' | 'GESTURE' | 'HAND' | 'AI' | 'ERROR';

export interface AnalysisEvent {
  id?: number;
  type: 'movement_event';
  session_id: string;
  timestamp_ms: number;
  person_id?: string;
  category: EventCategory;
  action: string;
  description: string;
  confidence?: number;
  source: string;
  severity: 'info' | 'warning' | 'alert';
}

export interface FrameStats {
  type: 'frame_stats';
  session_id: string;
  fps: number;
  latency_ms: number;
  person_count: number;
  face_count: number;
  hand_count: number;
  processing_mode: string;
  device: string;
  timestamp_ms: number;
}

export interface VideoPosition {
  type: 'video_position';
  session_id: string;
  current_s: number;
  duration_s: number;
  progress: number;
}

export interface WSMessage {
  type: string;
  session_id?: string;
  status?: string;
  message?: string;
  models_loaded?: Record<string, boolean>;
  [key: string]: unknown;
}

export interface UploadedFile {
  serverPath: string;
  filename: string;
  size_bytes: number;
  duration_seconds?: number;
}

export interface AppSettings {
  processingMode: ProcessingMode;
  enableFace: boolean;
  enablePose: boolean;
  enableHands: boolean;
  enableAI: boolean;
  cameraIndex: number;
  showBoundingBoxes: boolean;
  showSkeleton: boolean;
  showFaceLandmarks: boolean;
  showHandLandmarks: boolean;
  showConfidence: boolean;
  terminalAutoScroll: boolean;
  terminalMaxLines: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  processingMode: 'balanced',
  enableFace: true,
  enablePose: true,
  enableHands: true,
  enableAI: false,
  cameraIndex: 0,
  showBoundingBoxes: true,
  showSkeleton: true,
  showFaceLandmarks: false,
  showHandLandmarks: true,
  showConfidence: true,
  terminalAutoScroll: true,
  terminalMaxLines: 500,
};

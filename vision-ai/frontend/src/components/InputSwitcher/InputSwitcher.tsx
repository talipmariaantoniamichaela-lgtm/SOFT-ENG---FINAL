import React from 'react';
import { Camera, Upload, ChevronDown } from 'lucide-react';
import type { InputMode } from '../../types';

interface InputSwitcherProps {
  mode: InputMode;
  onModeChange: (mode: InputMode) => void;
  cameras: { index: number; label: string }[];
  selectedCamera: number;
  onCameraChange: (index: number) => void;
  uploadedFile?: { filename: string; duration_seconds?: number } | null;
  onFileSelect: (file: File) => void;
  isDragging: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
  uploading: boolean;
}

export const InputSwitcher: React.FC<InputSwitcherProps> = ({
  mode, onModeChange, cameras, selectedCamera, onCameraChange,
  uploadedFile, onFileSelect, isDragging, onDragOver, onDragLeave, onDrop,
  uploading,
}) => {
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  return (
    <div className="flex items-center gap-3">
      {/* Segmented control */}
      <div className="flex bg-bg-card border border-border-subtle rounded p-0.5 gap-0.5">
        <button
          onClick={() => onModeChange('live_camera')}
          className={mode === 'live_camera' ? 'seg-active' : 'seg-inactive'}
          style={{ minWidth: 110 }}
        >
          <span className="flex items-center justify-center gap-1.5">
            <Camera size={11} />
            Live Camera
          </span>
        </button>
        <button
          onClick={() => onModeChange('uploaded_video')}
          className={mode === 'uploaded_video' ? 'seg-active' : 'seg-inactive'}
          style={{ minWidth: 110 }}
        >
          <span className="flex items-center justify-center gap-1.5">
            <Upload size={11} />
            Upload Video
          </span>
        </button>
      </div>

      {/* Mode-specific controls */}
      {mode === 'live_camera' && cameras.length > 0 && (
        <div className="relative">
          <select
            value={selectedCamera}
            onChange={(e) => onCameraChange(Number(e.target.value))}
            className="input-base pr-6 appearance-none cursor-pointer"
          >
            {cameras.map((c) => (
              <option key={c.index} value={c.index}>{c.label}</option>
            ))}
          </select>
          <ChevronDown size={10} className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
        </div>
      )}

      {mode === 'uploaded_video' && (
        <div className="flex items-center gap-2">
          {uploadedFile ? (
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded bg-bg-card border border-green-accent/20">
              <div className="w-1.5 h-1.5 rounded-full bg-green-accent" />
              <span className="text-xs text-text-primary truncate max-w-[200px]">{uploadedFile.filename}</span>
              {uploadedFile.duration_seconds && (
                <span className="text-xs text-text-muted">
                  {Math.floor(uploadedFile.duration_seconds / 60)}:{String(Math.round(uploadedFile.duration_seconds % 60)).padStart(2, '0')}
                </span>
              )}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-[10px] text-cyan-accent hover:text-cyan-300"
              >
                Change
              </button>
            </div>
          ) : (
            <div
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`flex items-center gap-2 px-3 py-1.5 rounded border cursor-pointer
                transition-all duration-150 text-xs
                ${isDragging
                  ? 'border-cyan-accent bg-cyan-accent/5 text-cyan-accent'
                  : 'border-dashed border-border-accent text-text-muted hover:border-cyan-dim hover:text-text-primary'
                }
                ${uploading ? 'opacity-50 cursor-wait' : ''}`}
            >
              <Upload size={11} />
              {uploading ? 'Uploading...' : isDragging ? 'Drop video here' : 'Drop video or click to browse'}
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.webm,.avi,.mov,.mkv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFileSelect(file);
              e.target.value = '';
            }}
          />
        </div>
      )}
    </div>
  );
};

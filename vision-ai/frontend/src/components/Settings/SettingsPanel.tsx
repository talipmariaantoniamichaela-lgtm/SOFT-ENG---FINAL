import React from 'react';
import { X, Save } from 'lucide-react';
import type { AppSettings, ProcessingMode } from '../../types';

interface SettingsPanelProps {
  settings: AppSettings;
  onUpdate: (settings: AppSettings) => void;
  onClose: () => void;
}

const SectionHeader: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="text-[10px] text-text-dim uppercase tracking-widest mb-2 mt-4 first:mt-0">{children}</p>
);

const Toggle: React.FC<{
  label: string;
  description?: string;
  value: boolean;
  onChange: (v: boolean) => void;
  id: string;
}> = ({ label, description, value, onChange, id }) => (
  <label htmlFor={id} className="flex items-center justify-between py-2 cursor-pointer group">
    <div>
      <div className="text-xs text-text-primary group-hover:text-text-primary">{label}</div>
      {description && <div className="text-[10px] text-text-dim">{description}</div>}
    </div>
    <div
      onClick={() => onChange(!value)}
      className={`relative w-8 h-4 rounded-full transition-colors duration-200 cursor-pointer
        ${value ? 'bg-cyan-accent' : 'bg-bg-card border border-border-subtle'}`}
    >
      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform duration-200
        ${value ? 'translate-x-4' : 'translate-x-0.5'}`} />
    </div>
  </label>
);

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ settings, onUpdate, onClose }) => {
  const [local, setLocal] = React.useState<AppSettings>({ ...settings });

  const set = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) =>
    setLocal((prev) => ({ ...prev, [key]: value }));

  const save = () => {
    onUpdate(local);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-[480px] max-h-[85vh] bg-bg-secondary border border-border-subtle rounded-xl shadow-2xl flex flex-col animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border-subtle">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">Settings</h2>
            <p className="text-[11px] text-text-muted">Configure VISION AI behavior</p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5">
            <X size={14} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {/* Processing */}
          <SectionHeader>Processing Mode</SectionHeader>
          <div className="flex gap-1 p-0.5 bg-bg-card border border-border-subtle rounded mb-4">
            {(['performance', 'balanced', 'accuracy'] as ProcessingMode[]).map((m) => (
              <button
                key={m}
                onClick={() => set('processingMode', m)}
                className={`flex-1 py-1.5 text-xs font-medium rounded transition-colors capitalize
                  ${local.processingMode === m ? 'bg-cyan-accent text-bg-primary' : 'text-text-muted hover:text-text-primary'}`}
              >
                {m}
              </button>
            ))}
          </div>

          {/* Detection */}
          <SectionHeader>Detection</SectionHeader>
          <div className="bg-bg-card border border-border-subtle rounded-lg px-3 divide-y divide-border-subtle/40">
            <Toggle id="enableFace" label="Face Tracking" description="Detect head orientation and facial landmarks" value={local.enableFace} onChange={(v) => set('enableFace', v)} />
            <Toggle id="enablePose" label="Pose Tracking" description="Skeletal landmark detection and body analysis" value={local.enablePose} onChange={(v) => set('enablePose', v)} />
            <Toggle id="enableHands" label="Hand Tracking" description="Hand landmark detection and gesture recognition" value={local.enableHands} onChange={(v) => set('enableHands', v)} />
            <Toggle id="enableAI" label="AI Descriptions (Experimental)" description="Optional intelligent natural-language action summaries" value={local.enableAI} onChange={(v) => set('enableAI', v)} />
          </div>

          {/* Display */}
          <SectionHeader>Display</SectionHeader>
          <div className="bg-bg-card border border-border-subtle rounded-lg px-3 divide-y divide-border-subtle/40">
            <Toggle id="showBoundingBoxes" label="Bounding Boxes" value={local.showBoundingBoxes} onChange={(v) => set('showBoundingBoxes', v)} />
            <Toggle id="showSkeleton" label="Skeletal Overlay" value={local.showSkeleton} onChange={(v) => set('showSkeleton', v)} />
            <Toggle id="showFaceLandmarks" label="Facial Landmarks" value={local.showFaceLandmarks} onChange={(v) => set('showFaceLandmarks', v)} />
            <Toggle id="showHandLandmarks" label="Hand Landmarks" value={local.showHandLandmarks} onChange={(v) => set('showHandLandmarks', v)} />
            <Toggle id="showConfidence" label="Confidence Scores" value={local.showConfidence} onChange={(v) => set('showConfidence', v)} />
          </div>

          {/* Terminal */}
          <SectionHeader>Terminal</SectionHeader>
          <div className="bg-bg-card border border-border-subtle rounded-lg px-3 divide-y divide-border-subtle/40">
            <Toggle id="terminalAutoScroll" label="Auto-scroll" description="Follow the latest events" value={local.terminalAutoScroll} onChange={(v) => set('terminalAutoScroll', v)} />
            <div className="flex items-center justify-between py-2">
              <span className="text-xs text-text-primary">Max Lines</span>
              <input
                type="number"
                value={local.terminalMaxLines}
                onChange={(e) => set('terminalMaxLines', Math.max(100, parseInt(e.target.value) || 500))}
                className="input-base w-20 text-right"
                min={100}
                max={5000}
              />
            </div>
          </div>

          {/* Privacy notice */}
          <div className="mt-4 p-3 rounded-lg bg-bg-card border border-border-subtle">
            <p className="text-[11px] text-text-secondary leading-relaxed">
              <span className="text-text-primary font-medium">Privacy:</span> All video processing and AI inference occurs locally on your device.
              No video frames, facial data, or movement events are transmitted to any external service.
              Uploaded files are stored temporarily in the local <code className="text-cyan-accent text-[10px]">uploads/</code> directory.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 px-5 py-3 border-t border-border-subtle">
          <button onClick={onClose} className="btn-secondary">Cancel</button>
          <button onClick={save} className="btn-primary gap-1.5">
            <Save size={12} />
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

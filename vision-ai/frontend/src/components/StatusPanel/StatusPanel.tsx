import React from 'react';
import { Users, Eye, Hand, Activity, Cpu, Zap, CheckCircle2, XCircle } from 'lucide-react';

interface StatusPanelProps {
  personCount: number;
  faceCount: number;
  handCount: number;
  fps: number;
  latencyMs: number;
  processingMode: string;
  device: string;
  modelsLoaded: Record<string, boolean>;
  currentPosture: string;
  eventCount: number;
}

const StatRow: React.FC<{ label: string; value: React.ReactNode; icon: React.ReactNode }> = ({ label, value, icon }) => (
  <div className="flex items-center justify-between py-1.5 border-b border-border-subtle/40 last:border-0">
    <div className="flex items-center gap-1.5 text-text-muted text-xs">
      {icon}
      {label}
    </div>
    <div className="text-xs font-medium text-text-primary">{value}</div>
  </div>
);

export const StatusPanel: React.FC<StatusPanelProps> = ({
  personCount, faceCount, handCount, fps, latencyMs,
  processingMode, device, modelsLoaded, currentPosture, eventCount,
}) => {
  return (
    <div className="panel flex flex-col h-full overflow-hidden">
      <div className="panel-header shrink-0">
        <span className="panel-label">
          <Activity size={12} className="text-cyan-accent" />
          Detection Status
        </span>
        <div className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
          device === 'cuda' ? 'bg-green-accent/10 text-green-400' : 'bg-bg-card text-text-dim'
        }`}>
          <Cpu size={9} />
          {device.toUpperCase()}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2">
        {/* Detection counts */}
        <div className="mb-3">
          <p className="text-[10px] text-text-dim uppercase tracking-widest mb-1.5">Detections</p>
          <StatRow
            label="Persons"
            icon={<Users size={10} />}
            value={
              <span className={personCount > 0 ? 'text-green-400' : 'text-text-dim'}>
                {personCount}
              </span>
            }
          />
          <StatRow
            label="Faces"
            icon={<Eye size={10} />}
            value={
              <span className={faceCount > 0 ? 'text-cyan-400' : 'text-text-dim'}>
                {faceCount}
              </span>
            }
          />
          <StatRow
            label="Hands"
            icon={<Hand size={10} />}
            value={
              <span className={handCount > 0 ? 'text-pink-400' : 'text-text-dim'}>
                {handCount}
              </span>
            }
          />
          {currentPosture && currentPosture !== 'Unknown' && (
            <StatRow
              label="Posture"
              icon={<Activity size={10} />}
              value={<span className="text-amber-400">{currentPosture}</span>}
            />
          )}
        </div>

        {/* Performance */}
        <div className="mb-3">
          <p className="text-[10px] text-text-dim uppercase tracking-widest mb-1.5">Performance</p>
          <StatRow
            label="FPS"
            icon={<Zap size={10} />}
            value={
              <span className={fps >= 20 ? 'text-green-400' : fps >= 10 ? 'text-amber-400' : 'text-red-400'}>
                {fps.toFixed(0)}
              </span>
            }
          />
          <StatRow
            label="Latency"
            icon={<Activity size={10} />}
            value={
              <span className={latencyMs < 80 ? 'text-green-400' : latencyMs < 200 ? 'text-amber-400' : 'text-red-400'}>
                {latencyMs.toFixed(0)}ms
              </span>
            }
          />
          <StatRow
            label="Mode"
            icon={<Cpu size={10} />}
            value={
              <span className="capitalize text-text-secondary">{processingMode}</span>
            }
          />
          <StatRow
            label="Events"
            icon={<Activity size={10} />}
            value={<span className="text-text-secondary">{eventCount.toLocaleString()}</span>}
          />
        </div>

        {/* Models */}
        <div>
          <p className="text-[10px] text-text-dim uppercase tracking-widest mb-1.5">AI Models</p>
          {Object.entries(modelsLoaded).map(([name, loaded]) => (
            <div key={name} className="flex items-center justify-between py-1">
              <span className="text-xs text-text-muted capitalize">{name} Landmarker</span>
              {loaded
                ? <CheckCircle2 size={11} className="text-green-accent" />
                : <XCircle size={11} className="text-text-dim" />
              }
            </div>
          ))}
        </div>

        {/* Privacy notice */}
        <div className="mt-3 p-2 rounded bg-bg-card border border-border-subtle/40">
          <p className="text-[10px] text-text-dim leading-relaxed">
            All processing occurs locally on this device. No video or facial data is transmitted externally.
          </p>
        </div>
      </div>
    </div>
  );
};

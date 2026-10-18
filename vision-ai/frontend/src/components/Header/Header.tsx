import React from 'react';
import { Settings, Activity, Cpu, Shield } from 'lucide-react';
import type { SessionStatus } from '../../types';
import type { WSStatus } from '../../hooks/useAnalysisWS';

interface HeaderProps {
  sessionStatus: SessionStatus;
  wsStatus: WSStatus;
  backendOnline: boolean;
  onSettingsClick: () => void;
}

const STATUS_CONFIG: Record<SessionStatus, { label: string; color: string; dot: string }> = {
  idle:         { label: 'Idle',          color: 'text-text-muted',    dot: 'bg-text-dim' },
  initializing: { label: 'Initializing',  color: 'text-amber-accent',  dot: 'bg-amber-accent animate-pulse' },
  running:      { label: 'Analyzing',     color: 'text-green-accent',  dot: 'bg-green-accent animate-pulse' },
  paused:       { label: 'Paused',        color: 'text-amber-accent',  dot: 'bg-amber-accent' },
  stopped:      { label: 'Stopped',       color: 'text-text-muted',    dot: 'bg-text-dim' },
  error:        { label: 'Error',         color: 'text-red-accent',    dot: 'bg-red-accent animate-pulse' },
};

export const Header: React.FC<HeaderProps> = ({
  sessionStatus, wsStatus, backendOnline, onSettingsClick,
}) => {
  const sc = STATUS_CONFIG[sessionStatus];

  return (
    <header className="h-12 flex items-center justify-between px-4 border-b border-border-subtle bg-bg-secondary shrink-0">
      {/* Left — Logo + Name */}
      <div className="flex items-center gap-3">
        {/* SVG Logo */}
        <div className="flex items-center justify-center w-7 h-7 rounded bg-cyan-accent/10 border border-cyan-accent/20">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" className="text-cyan-accent">
            <circle cx="12" cy="6" r="3" fill="currentColor" opacity="0.9"/>
            <path d="M12 9 L9 14 L15 14 Z" fill="currentColor" opacity="0.7"/>
            <path d="M9 14 L6 19" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            <path d="M15 14 L18 19" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            <path d="M9 14 L6 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            <path d="M15 14 L18 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-bold text-text-primary tracking-tight">VISION</span>
            <span className="text-sm font-bold text-cyan-accent tracking-tight">AI</span>
          </div>
          <p className="text-[10px] text-text-dim leading-none">Intelligent Human Movement Analysis</p>
        </div>
      </div>

      {/* Center — Status indicators */}
      <div className="flex items-center gap-5">
        {/* Session status */}
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${sc.dot}`} />
          <span className={`text-xs ${sc.color}`}>{sc.label}</span>
        </div>

        <div className="w-px h-4 bg-border-subtle" />

        {/* Backend status */}
        <div className="flex items-center gap-1.5">
          <Cpu size={11} className={backendOnline ? 'text-green-accent' : 'text-red-accent'} />
          <span className={`text-xs ${backendOnline ? 'text-text-muted' : 'text-red-accent'}`}>
            {backendOnline ? 'Backend Online' : 'Backend Offline'}
          </span>
        </div>

        <div className="w-px h-4 bg-border-subtle" />

        {/* WS status */}
        <div className="flex items-center gap-1.5">
          <Activity size={11} className={
            wsStatus === 'connected' ? 'text-cyan-accent animate-pulse' :
            wsStatus === 'connecting' ? 'text-amber-accent animate-spin' : 'text-text-dim'
          } />
          <span className="text-xs text-text-muted">
            {wsStatus === 'connected' ? 'WS Connected' :
             wsStatus === 'connecting' ? 'WS Connecting' :
             sessionStatus === 'idle' ? 'WS Standby' : 'WS Disconnected'}
          </span>
        </div>
      </div>

      {/* Right — Actions */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 px-2 py-1 rounded bg-bg-card border border-border-subtle">
          <Shield size={10} className="text-text-dim" />
          <span className="text-[10px] text-text-dim">Local Processing</span>
        </div>
        <button
          onClick={onSettingsClick}
          className="btn-ghost p-1.5 rounded hover:bg-bg-card"
          title="Settings"
        >
          <Settings size={14} className="text-text-muted" />
        </button>
      </div>
    </header>
  );
};

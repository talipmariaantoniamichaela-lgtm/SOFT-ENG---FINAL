import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  Terminal, ChevronDown, ChevronUp, Search, Filter, Trash2,
  Download, Pause, Play, X
} from 'lucide-react';
import type { AnalysisEvent, EventCategory } from '../../types';
import { api } from '../../services/api';

const CATEGORY_COLORS: Record<string, string> = {
  SYSTEM:  'text-text-muted',
  DETECT:  'text-blue-400',
  FACE:    'text-cyan-400',
  ACTION:  'text-green-400',
  POSTURE: 'text-amber-400',
  GESTURE: 'text-purple-400',
  HAND:    'text-pink-400',
  AI:      'text-indigo-400',
  ERROR:   'text-red-400',
};

const BADGE_CLS: Record<string, string> = {
  SYSTEM:  'badge-system',
  DETECT:  'badge-detect',
  FACE:    'badge-face',
  ACTION:  'badge-action',
  POSTURE: 'badge-posture',
  GESTURE: 'badge-gesture',
  HAND:    'badge-hand',
  AI:      'badge-ai',
  ERROR:   'badge-error',
};

interface AnalysisTerminalProps {
  events: AnalysisEvent[];
  autoScroll: boolean;
  onToggleAutoScroll: () => void;
  onClear: () => void;
  sessionId: string | null;
}

const ALL_CATEGORIES: EventCategory[] = ['SYSTEM', 'DETECT', 'FACE', 'ACTION', 'POSTURE', 'GESTURE', 'HAND', 'AI', 'ERROR'];

function formatTimestamp(ms: number): string {
  const s = ms / 1000;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms2 = ms % 1000;
  if (h > 0) return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(ms2).padStart(3, '0')}`;
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(ms2).padStart(3, '0')}`;
}

export const AnalysisTerminal: React.FC<AnalysisTerminalProps> = ({
  events, autoScroll, onToggleAutoScroll, onClear, sessionId,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<EventCategory | 'ALL'>('ALL');
  const [showFilter, setShowFilter] = useState(false);
  const [expanded, setExpanded] = useState(true);

  // Filtered events
  const filtered = useMemo(() => {
    let ev = events;
    if (filter !== 'ALL') ev = ev.filter(e => e.category === filter);
    if (search.trim()) {
      const q = search.toLowerCase();
      ev = ev.filter(e =>
        e.description.toLowerCase().includes(q) ||
        e.category.toLowerCase().includes(q) ||
        (e.person_id || '').toLowerCase().includes(q)
      );
    }
    return ev;
  }, [events, filter, search]);

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filtered.length, autoScroll]);

  const handleExport = useCallback((format: 'json' | 'csv') => {
    if (sessionId) api.exportEvents(sessionId, format);
  }, [sessionId]);

  return (
    <div className="panel flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="panel-header shrink-0">
        <div className="flex items-center gap-2">
          <Terminal size={12} className="text-cyan-accent" />
          <span className="panel-label">Analysis Terminal</span>
          <span className="text-[10px] text-text-dim bg-bg-card px-1.5 py-0.5 rounded">
            {filtered.length}/{events.length}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {/* Auto-scroll toggle */}
          <button onClick={onToggleAutoScroll} className="btn-ghost px-1.5 py-1" title={autoScroll ? 'Pause scroll' : 'Resume scroll'}>
            {autoScroll ? <Pause size={10} /> : <Play size={10} />}
          </button>

          {/* Filter */}
          <div className="relative">
            <button
              onClick={() => setShowFilter(!showFilter)}
              className={`btn-ghost px-1.5 py-1 ${filter !== 'ALL' ? 'text-cyan-accent' : ''}`}
              title="Filter by category"
            >
              <Filter size={10} />
            </button>
            {showFilter && (
              <div className="absolute right-0 top-full mt-1 bg-bg-card border border-border-subtle rounded shadow-xl z-50 py-1 min-w-[120px] animate-fade-in">
                {(['ALL', ...ALL_CATEGORIES] as const).map((cat) => (
                  <button
                    key={cat}
                    onClick={() => { setFilter(cat as EventCategory | 'ALL'); setShowFilter(false); }}
                    className={`w-full text-left px-3 py-1 text-xs hover:bg-bg-tertiary transition-colors
                      ${filter === cat ? 'text-cyan-accent' : 'text-text-secondary'}`}
                  >
                    {cat === 'ALL' ? 'All Categories' : cat}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Export */}
          <div className="relative group">
            <button className="btn-ghost px-1.5 py-1" title="Export events" disabled={!sessionId}>
              <Download size={10} />
            </button>
            <div className="absolute right-0 top-full mt-1 bg-bg-card border border-border-subtle rounded shadow-xl z-50 py-1 hidden group-hover:block min-w-[80px]">
              <button onClick={() => handleExport('json')} className="w-full text-left px-3 py-1 text-xs text-text-secondary hover:bg-bg-tertiary">JSON</button>
              <button onClick={() => handleExport('csv')} className="w-full text-left px-3 py-1 text-xs text-text-secondary hover:bg-bg-tertiary">CSV</button>
            </div>
          </div>

          {/* Clear */}
          <button onClick={onClear} className="btn-ghost px-1.5 py-1 hover:text-red-400" title="Clear terminal">
            <Trash2 size={10} />
          </button>

          {/* Collapse */}
          <button onClick={() => setExpanded(!expanded)} className="btn-ghost px-1.5 py-1">
            {expanded ? <ChevronDown size={10} /> : <ChevronUp size={10} />}
          </button>
        </div>
      </div>

      {expanded && (
        <>
          {/* Search bar */}
          <div className="px-2 py-1.5 border-b border-border-subtle shrink-0">
            <div className="relative">
              <Search size={10} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-dim" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search events..."
                className="input-base w-full pl-6 py-1 text-[11px]"
              />
              {search && (
                <button onClick={() => setSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-text-dim hover:text-text-primary">
                  <X size={10} />
                </button>
              )}
            </div>
          </div>

          {/* Events list */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto font-mono text-[11px] min-h-0">
            {filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-2 text-text-dim">
                <Terminal size={20} />
                <p className="text-xs">
                  {events.length === 0
                    ? 'Start analysis to see events'
                    : 'No events match the current filter'}
                </p>
              </div>
            ) : (
              <div className="py-1">
                {filtered.map((ev, i) => (
                  <div
                    key={i}
                    className={`flex items-start gap-2 px-3 py-0.5 hover:bg-bg-card/50 transition-colors
                      ${ev.category === 'ERROR' ? 'bg-red-accent/5' : ''}`}
                  >
                    <span className="text-text-dim shrink-0 tabular-nums leading-5">
                      [{formatTimestamp(ev.timestamp_ms)}]
                    </span>
                    <span className={`shrink-0 ${BADGE_CLS[ev.category] || 'badge-system'} mt-0.5`}>
                      {ev.category}
                    </span>
                    <span className={`flex-1 leading-5 ${CATEGORY_COLORS[ev.category] || 'text-text-muted'}`}>
                      {ev.description}
                      {ev.confidence !== undefined && ev.confidence !== null && (
                        <span className="ml-1.5 text-text-dim">
                          ({(ev.confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

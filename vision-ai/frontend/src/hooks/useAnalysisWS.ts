// WebSocket hook for VISION AI real-time analysis
import { useEffect, useRef, useCallback, useState } from 'react';
import { WS_BASE } from '../services/api';
import type { AnalysisEvent, FrameStats, VideoPosition, WSMessage } from '../types';

export type WSStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface UseAnalysisWSOptions {
  sessionId: string | null;
  onEvent?: (event: AnalysisEvent) => void;
  onStats?: (stats: FrameStats) => void;
  onVideoPosition?: (pos: VideoPosition) => void;
  onStatusChange?: (status: string, message: string, meta?: Record<string, unknown>) => void;
}

export function useAnalysisWS({
  sessionId,
  onEvent,
  onStats,
  onVideoPosition,
  onStatusChange,
}: UseAnalysisWSOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [wsStatus, setWsStatus] = useState<WSStatus>('disconnected');
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const pendingMessagesRef = useRef<object[]>([]);
  const mountedRef = useRef(true);

  useEffect(() => {
    return () => { mountedRef.current = false; };
  }, []);

  const connect = useCallback(() => {
    if (!sessionId) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setWsStatus('connecting');
    const ws = new WebSocket(`${WS_BASE}/ws/analysis/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setWsStatus('connected');
      // Flush any queued commands (e.g. start)
      while (pendingMessagesRef.current.length > 0) {
        const msg = pendingMessagesRef.current.shift();
        if (msg) ws.send(JSON.stringify(msg));
      }
    };

    ws.onmessage = (ev) => {
      if (!mountedRef.current) return;
      try {
        const msg = JSON.parse(ev.data) as WSMessage;
        if (msg.type === 'movement_event') {
          onEvent?.(msg as unknown as AnalysisEvent);
        } else if (msg.type === 'frame_stats') {
          onStats?.(msg as unknown as FrameStats);
        } else if (msg.type === 'video_position') {
          onVideoPosition?.(msg as unknown as VideoPosition);
        } else if (msg.type === 'session_status') {
          onStatusChange?.(msg.status || '', msg.message || '', msg as Record<string, unknown>);
        } else if (msg.type === 'ping') {
          ws.send(JSON.stringify({ cmd: 'ping' }));
        }
      } catch (e) {
        console.error('[WS] Parse error:', e);
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setWsStatus('error');
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setWsStatus('disconnected');
    };
  }, [sessionId, onEvent, onStats, onVideoPosition, onStatusChange]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    pendingMessagesRef.current = [];
    wsRef.current?.close();
    wsRef.current = null;
    setWsStatus('disconnected');
  }, []);

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    } else {
      pendingMessagesRef.current.push(msg);
    }
  }, []);

  const sendStart = useCallback((config?: object) =>
    send({ cmd: 'start', config: config || {} }), [send]);

  const sendStop = useCallback(() => send({ cmd: 'stop' }), [send]);
  const sendPause = useCallback(() => send({ cmd: 'pause' }), [send]);
  const sendResume = useCallback(() => send({ cmd: 'resume' }), [send]);
  const sendSeek = useCallback((progress: number) =>
    send({ cmd: 'seek', progress }), [send]);

  useEffect(() => {
    if (sessionId) {
      connect();
    }
    return () => {
      wsRef.current?.close();
    };
  }, [sessionId]);

  return { wsStatus, connect, disconnect, sendStart, sendStop, sendPause, sendResume, sendSeek };
}

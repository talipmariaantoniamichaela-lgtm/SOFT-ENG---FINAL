// API service for VISION AI backend
const BASE = 'http://localhost:8080/api';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.error || res.statusText);
  }
  return res.json();
}

export const api = {
  health: () => apiFetch<{ status: string; version: string; device: string; uptime_seconds: number }>('/health'),

  startSession: (config: object) =>
    apiFetch<{ session_id: string; status: string }>('/session/start', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),

  stopSession: (id: string) =>
    apiFetch<{ status: string }>(`/session/${id}/stop`, { method: 'POST' }),

  resetSession: (id: string) =>
    apiFetch<{ status: string }>(`/session/${id}/reset`, { method: 'POST' }),

  deleteSession: (id: string) =>
    apiFetch<{ status: string }>(`/session/${id}`, { method: 'DELETE' }),

  getEvents: (id: string, page = 1, pageSize = 100, category?: string) =>
    apiFetch<{ events: unknown[]; total: number; pages: number }>(
      `/session/${id}/events?page=${page}&page_size=${pageSize}${category ? `&category=${category}` : ''}`
    ),

  exportEvents: (id: string, format: 'json' | 'csv') => {
    const a = document.createElement('a');
    a.href = `${BASE}/session/${id}/export?format=${format}`;
    a.download = `session_${id.slice(0, 8)}_events.${format}`;
    a.click();
  },

  uploadVideo: async (file: File): Promise<{ session_id: string; filename: string; size_bytes: number; duration_seconds?: number }> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/video/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  },
};

export const WS_BASE = 'ws://localhost:8080';

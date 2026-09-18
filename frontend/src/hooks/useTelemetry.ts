import { useEffect, useRef, useState, useCallback } from 'react';
import type { TelemetryData } from '../types/api';
import { FALLBACK_TELEMETRY } from '../mock/fallbackData';
import { API } from '../services/api';

const POLL_INTERVAL_MS = 40; // 25 Hz

export function useTelemetry() {
  const [data, setData] = useState<TelemetryData>(FALLBACK_TELEMETRY);
  const [connected, setConnected] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    try {
      const res = await fetch(`${API.TELEMETRY}?t=${Date.now()}`, {
        signal: abortRef.current.signal,
      });
      if (res.ok) {
        const json: TelemetryData = await res.json();
        // Guard: only update if frame_id is present (a real response)
        if (json && json.step !== undefined) {
          setData(json);
          setConnected(true);
        }
      } else {
        setConnected(false);
      }
    } catch {
      setConnected(false);
    } finally {
      timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
    }
  }, []);

  useEffect(() => {
    poll();
    return () => {
      abortRef.current?.abort();
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [poll]);

  return { data, connected };
}

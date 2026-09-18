import { useEffect, useRef, useState, useCallback } from 'react';
import type { DigitalTwinData } from '../types/api';
import { FALLBACK_DIGITAL_TWIN } from '../mock/fallbackData';
import { API } from '../services/api';

const POLL_INTERVAL_MS = 100; // 10 Hz — 3D scene doesn't need full 25 Hz

export function useDigitalTwin() {
  const [data, setData] = useState<DigitalTwinData>(FALLBACK_DIGITAL_TWIN);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API.DIGITAL_TWIN}?t=${Date.now()}`);
      if (res.ok) {
        const json: DigitalTwinData = await res.json();
        if (json && json.container) {
          setData(json);
        }
      }
    } catch {
      // keep previous data if request fails
    } finally {
      timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
    }
  }, []);

  useEffect(() => {
    poll();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [poll]);

  return { data };
}

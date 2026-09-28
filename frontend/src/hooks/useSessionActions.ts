import { useEffect, useRef, useState, useCallback } from 'react';
import type { SessionActionsData } from '../types/api';
import { FALLBACK_SESSION } from '../mock/fallbackData';
import { API } from '../services/api';

const POLL_INTERVAL_MS = 2000; // 0.5 Hz — session actions change slowly

export function useSessionActions() {
  const [data, setData] = useState<SessionActionsData>(FALLBACK_SESSION);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API.SESSION_ACTIONS}?t=${Date.now()}`);
      if (res.ok) {
        const json: SessionActionsData = await res.json();
        if (json && json.session_id) {
          setData(json);
        }
      }
    } catch {
      // keep previous data
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

import { useEffect, useRef, useState, useCallback } from 'react';
import type { TelemetryData } from '../types/api';
import { FALLBACK_TELEMETRY } from '../mock/fallbackData';
import { API } from '../services/api';

const POLL_INTERVAL_MS = 40; // 25 Hz
const MAX_FAIL_COUNT = 5;    // Require 5 consecutive failures (~500ms) before declaring offline

function shallowTelemetryEqual(a: TelemetryData, b: TelemetryData): boolean {
  if (a === b) return true;
  return (
    a.frame_id === b.frame_id &&
    a.step === b.step &&
    a.step_name === b.step_name &&
    a.is_step_correct === b.is_step_correct &&
    a.step_verdict === b.step_verdict &&
    a.what_i_am_doing === b.what_i_am_doing &&
    a.what_i_have_to_do === b.what_i_have_to_do &&
    a.anomaly === b.anomaly &&
    a.transition_event === b.transition_event &&
    a.fps === b.fps &&
    a.latency_ms === b.latency_ms &&
    a.source_type === b.source_type
  );
}

export function useTelemetry() {
  const [data, setData] = useState<TelemetryData>(FALLBACK_TELEMETRY);
  const [connected, setConnected] = useState(false);
  const failCountRef = useRef(0);
  const inFlightRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    if (inFlightRef.current) {
      // Avoid overlapping requests that cause state thrashing
      timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
      return;
    }

    inFlightRef.current = true;
    try {
      const res = await fetch(`${API.TELEMETRY}?t=${Date.now()}`);
      if (res.ok) {
        const json: TelemetryData = await res.json();
        if (json && json.step !== undefined) {
          failCountRef.current = 0;
          setConnected(prev => (!prev ? true : prev));
          setData(prev => (shallowTelemetryEqual(prev, json) ? prev : json));
        }
      } else {
        failCountRef.current += 1;
        if (failCountRef.current >= MAX_FAIL_COUNT) {
          setConnected(prev => (prev ? false : prev));
        }
      }
    } catch {
      failCountRef.current += 1;
      if (failCountRef.current >= MAX_FAIL_COUNT) {
        setConnected(prev => (prev ? false : prev));
      }
    } finally {
      inFlightRef.current = false;
      timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
    }
  }, []);

  useEffect(() => {
    poll();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [poll]);

  return { data, connected };
}

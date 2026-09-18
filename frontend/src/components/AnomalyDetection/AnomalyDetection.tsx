import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, ArrowDown, CheckCircle2 } from 'lucide-react';
import { useSessionActions } from '../../hooks/useSessionActions';
import type { TelemetryData } from '../../types/api';

interface AnomalyDetectionProps {
  telemetry: TelemetryData;
}

export interface AnomalyIncident {
  id: string;
  timestamp: string;
  code: string;
  step: string;
  verdict: string;
  severity: 'CRITICAL' | 'WARNING' | 'NOMINAL';
  message: string;
}

let incidentSeq = 0;
function makeId() { return `inc-${++incidentSeq}`; }
function nowHHMMSS() { return new Date().toTimeString().slice(0, 8); }

export function AnomalyDetection({ telemetry }: AnomalyDetectionProps) {
  const { data: session } = useSessionActions();
  const anomalyActive = Boolean(telemetry.anomaly && telemetry.anomaly !== 'NONE');
  const [lastAnomaly, setLastAnomaly] = useState<string>('NONE');
  const [eventTime, setEventTime] = useState<string>('');
  const [speaking, setSpeaking] = useState(false);
  const synthRef = useRef<SpeechSynthesisUtterance | null>(null);

  const [incidents, setIncidents] = useState<AnomalyIncident[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const prevSessionActionsCount = useRef<number>(0);

  // Ingest existing historical anomalies from session_actions on load
  useEffect(() => {
    if (!session.timeline || session.timeline.length === 0) return;
    if (session.timeline.length <= prevSessionActionsCount.current) return;

    const newActions = session.timeline.slice(prevSessionActionsCount.current);
    prevSessionActionsCount.current = session.timeline.length;

    const newIncidents: AnomalyIncident[] = [];
    for (const act of newActions) {
      if (act.compliance && act.compliance !== 'NOMINAL') {
        newIncidents.push({
          id: makeId(),
          timestamp: nowHHMMSS(),
          code: act.compliance,
          step: act.step_name,
          verdict: 'Procedural Deviation',
          severity: 'WARNING',
          message: `${act.step_name}: ${act.what_i_am_doing} (${act.duration_sec.toFixed(1)}s)`,
        });
      }
    }

    if (newIncidents.length > 0) {
      setIncidents(prev => [...prev, ...newIncidents].slice(-100));
    }
  }, [session.timeline]);

  // Trigger TTS and record real-time anomaly events on state transition
  useEffect(() => {
    const current = telemetry.anomaly;
    if (current === lastAnomaly) return;

    const time = nowHHMMSS();
    setLastAnomaly(current);

    if (current && current !== 'NONE') {
      setEventTime(time);
      speakAnomaly(current, setSpeaking, synthRef);

      const isCritical = current.includes('ERROR') || current.includes('VIOLATION') || current.includes('FAIL');
      const newInc: AnomalyIncident = {
        id: makeId(),
        timestamp: time,
        code: current,
        step: telemetry.step_name || `STEP ${telemetry.step}`,
        verdict: telemetry.step_verdict || 'Safety violation flagged',
        severity: isCritical ? 'CRITICAL' : 'WARNING',
        message: buildTTSText(current),
      };

      setIncidents(prev => [...prev, newInc].slice(-100));
    } else if (lastAnomaly !== 'NONE' && (!current || current === 'NONE')) {
      // Anomaly cleared
      const clearInc: AnomalyIncident = {
        id: makeId(),
        timestamp: time,
        code: 'RESOLVED',
        step: telemetry.step_name || `STEP ${telemetry.step}`,
        verdict: 'Nominal tracking restored',
        severity: 'NOMINAL',
        message: `Previous anomaly (${lastAnomaly.replace(/_/g, ' ')}) cleared.`,
      };
      setIncidents(prev => [...prev, clearInc].slice(-100));
    }
  }, [telemetry.anomaly, telemetry.step_name, telemetry.step, telemetry.step_verdict, lastAnomaly]);

  // Realtime auto-scroll to bottom
  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [incidents, autoScroll]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 35;
    if (atBottom !== autoScroll) {
      setAutoScroll(atBottom);
    }
  };

  const scrollToBottom = () => {
    setAutoScroll(true);
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  return (
    <div className={`panel anomaly-panel ${anomalyActive ? 'anomaly-active' : ''}`}>
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <AlertTriangle size={14} strokeWidth={2} style={{ color: 'var(--amber)' }} />
          <span className="panel-title">ANOMALY DETECTION</span>
          <span className={`anomaly-severity-badge ${anomalyActive ? '' : 'badge-ok'}`}>
            {anomalyActive ? 'WARNING' : 'NOMINAL'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            type="button"
            className={`autoscroll-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => {
              if (!autoScroll) {
                scrollToBottom();
              } else {
                setAutoScroll(false);
              }
            }}
            title={autoScroll ? "Realtime auto-scroll active. Click to pause." : "Auto-scroll paused. Click to resume."}
          >
            <span className="autoscroll-dot" />
            {autoScroll ? 'LIVE SCROLL' : 'SCROLL PAUSED'}
          </button>
        </div>
      </div>

      {/* Top Status Card */}
      <div className="anomaly-status-card">
        {!anomalyActive ? (
          <div className="anomaly-normal">
            <div className="anomaly-normal-left">
              <span className="status-dot-green" />
              <span className="anomaly-normal-text">ALL SYSTEMS NOMINAL</span>
            </div>
            <span className="anomaly-normal-sub">
              {telemetry.step_verdict ? telemetry.step_verdict : 'Zero safety violations'}
            </span>
          </div>
        ) : (
          <div className="anomaly-alert">
            <div className="anomaly-icon-row">
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <AlertTriangle size={13} className="anomaly-warn-icon" />
                <span className="anomaly-warn-label">ACTIVE FAULT</span>
              </div>
              <div className="anomaly-code">{telemetry.anomaly}</div>
            </div>

            <div className="anomaly-card-meta">
              <span className="anomaly-verdict">
                {telemetry.step_verdict || 'Procedural deviation detected'}
              </span>
              {eventTime && (
                <span className="anomaly-timestamp">Detected: {eventTime}</span>
              )}
            </div>

            {speaking && (
              <div className="tts-indicator">
                <WaveformIcon />
                <span className="tts-label">AUDIO ALERT BROADCASTING...</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Incident Feed Header */}
      <div className="anomaly-feed-header">
        <span>INCIDENT & DEVIATION FEED</span>
        <span className="badge-count font-mono">{incidents.length} EVENTS</span>
      </div>

      {/* Scrollable Incident Feed */}
      <div style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <div
          className="anomaly-scroll"
          ref={scrollRef}
          onScroll={handleScroll}
        >
          {incidents.length === 0 ? (
            <div className="anomaly-empty-state">
              <CheckCircle2 size={24} strokeWidth={1.5} style={{ color: 'var(--green)' }} />
              <div>No anomalies or safety deviations recorded</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Realtime telemetry is monitoring continuous pose & container limits</div>
            </div>
          ) : (
            incidents.map(inc => (
              <div
                key={inc.id}
                className={`anomaly-item item-${inc.severity.toLowerCase()}`}
              >
                <div className="anomaly-item-top">
                  <span className="anomaly-item-type">
                    {inc.code}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="anomaly-item-step">{inc.step}</span>
                    <span className="anomaly-item-time">{inc.timestamp}</span>
                  </div>
                </div>
                <div className="anomaly-item-desc">{inc.message}</div>
              </div>
            ))
          )}
        </div>

        {!autoScroll && incidents.length > 0 && (
          <button
            type="button"
            className="jump-bottom-btn"
            onClick={scrollToBottom}
          >
            <ArrowDown size={11} strokeWidth={2.5} /> Jump to Live
          </button>
        )}
      </div>
    </div>
  );
}

function WaveformIcon() {
  return (
    <svg className="waveform-icon" viewBox="0 0 32 14" fill="none">
      <rect x="0"  y="5"  width="3" height="4"  rx="1" fill="#f59e0b" className="wave-bar b1" />
      <rect x="5"  y="2"  width="3" height="10" rx="1" fill="#f59e0b" className="wave-bar b2" />
      <rect x="10" y="0"  width="3" height="14" rx="1" fill="#f59e0b" className="wave-bar b3" />
      <rect x="15" y="3"  width="3" height="8"  rx="1" fill="#f59e0b" className="wave-bar b4" />
      <rect x="20" y="5"  width="3" height="4"  rx="1" fill="#f59e0b" className="wave-bar b5" />
      <rect x="25" y="1"  width="3" height="12" rx="1" fill="#f59e0b" className="wave-bar b6" />
      <rect x="29" y="4"  width="3" height="6"  rx="1" fill="#f59e0b" className="wave-bar b7" />
    </svg>
  );
}

function speakAnomaly(
  anomaly: string,
  setSpeaking: (v: boolean) => void,
  synthRef: React.MutableRefObject<SpeechSynthesisUtterance | null>
) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();

  const text = buildTTSText(anomaly);
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.92;
  utter.pitch = 1.0;
  utter.volume = 1.0;
  utter.onstart = () => setSpeaking(true);
  utter.onend = () => setSpeaking(false);
  utter.onerror = () => setSpeaking(false);
  synthRef.current = utter;
  window.speechSynthesis.speak(utter);
}

function buildTTSText(anomaly: string): string {
  const map: Record<string, string> = {
    ERROR_SEQ: 'Warning. Unexpected action sequence detected.',
    WRONG_BOX: 'Warning. Wrong box detected.',
    EARLY_CLOSE: 'Warning. Container closed too early.',
    ROM_VIOLATION: 'Warning. Range of motion limits exceeded.',
    OUT_OF_ORDER: 'Warning. Step performed out of order.',
  };
  return map[anomaly] ?? `Warning. Anomaly detected: ${anomaly.replace(/_/g, ' ')}.`;
}

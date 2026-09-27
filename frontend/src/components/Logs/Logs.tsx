import { useEffect, useRef, useState } from 'react';
import { ArrowDown, Terminal } from 'lucide-react';
import { useSessionActions } from '../../hooks/useSessionActions';
import type { TelemetryData, LogEntry, LogLevel } from '../../types/api';
import { SmoothNumber, AnimatedText } from '../common/SmoothData';

interface LogsProps {
  telemetry: TelemetryData;
}

let logSeq = 0;
function makeId() { return `log-${++logSeq}`; }
function nowHHMMSS() { return new Date().toTimeString().slice(0, 8); }

export function Logs({ telemetry }: LogsProps) {
  const { data: session } = useSessionActions();
  const [logs, setLogs] = useState<LogEntry[]>([
    { id: makeId(), timestamp: nowHHMMSS(), level: 'INFO', message: 'BAS-HAR System frontend initialized.' },
  ]);

  const prevStepName = useRef<string>('');
  const prevAnomaly = useRef<string>('NONE');
  const prevTransition = useRef<string | null>(null);
  const prevSessionLen = useRef<number>(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Append log entry helper
  const append = (level: LogLevel, message: string) => {
    setLogs(prev => {
      const next = [...prev, { id: makeId(), timestamp: nowHHMMSS(), level, message }];
      // Keep last 300 entries
      return next.length > 300 ? next.slice(next.length - 300) : next;
    });
  };

  // Track telemetry changes → new log entries
  useEffect(() => {
    if (!telemetry.step_name) return;

    // Step transition
    if (telemetry.step_name !== prevStepName.current && prevStepName.current !== '') {
      append('INFO', `Step transitioned → ${telemetry.step_name} (${telemetry.what_i_am_doing || 'ACTIVE'})`);
    }
    prevStepName.current = telemetry.step_name;

    // Transition event (state machine emits this on milestone)
    if (telemetry.transition_event && telemetry.transition_event !== prevTransition.current) {
      const level: LogLevel = telemetry.anomaly !== 'NONE' ? 'WARNING' : 'SUCCESS';
      append(level, `Event: ${telemetry.transition_event}`);
      prevTransition.current = telemetry.transition_event;
    }

    // Anomaly
    const anom = telemetry.anomaly;
    if (anom !== prevAnomaly.current) {
      if (anom && anom !== 'NONE') {
        append('WARNING', `Anomaly detected: ${anom} — ${telemetry.step_verdict}`);
      } else if (prevAnomaly.current !== 'NONE') {
        append('INFO', 'Anomaly cleared — returning to nominal tracking');
      }
      prevAnomaly.current = anom;
    }
  }, [telemetry.step_name, telemetry.transition_event, telemetry.anomaly, telemetry.step_verdict, telemetry.what_i_am_doing]);

  // Ingest new session actions from /api/session_actions
  useEffect(() => {
    const timeline = session.timeline;
    if (timeline.length <= prevSessionLen.current) return;
    const newEntries = timeline.slice(prevSessionLen.current);
    prevSessionLen.current = timeline.length;
    for (const entry of newEntries) {
      const lvl: LogLevel = entry.compliance === 'NOMINAL' ? 'SUCCESS' : 'WARNING';
      append(lvl, `${entry.step_name}: ${entry.what_i_am_doing} (${entry.duration_sec.toFixed(1)}s)`);
    }
  }, [session.timeline]);

  // Realtime auto-scroll to bottom
  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [logs, autoScroll]);

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
    <div className="panel logs-panel">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Terminal size={14} strokeWidth={2} style={{ color: 'var(--accent)' }} />
          <span className="panel-title">SYSTEM LOGS</span>
          <span className="badge-count font-mono">
            <SmoothNumber value={logs.length} />
          </span>
        </div>
        <div className="logs-meta">
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
          <span className="panel-subtitle font-mono">{session.session_id}</span>
          <span className={`compliance-chip ${session.summary.anomalies_flagged > 0 ? 'chip-warn' : 'chip-ok'}`}>
            <AnimatedText inline>{session.summary.compliance_status}</AnimatedText>
          </span>
        </div>
      </div>

      {/* Session summary strip with smooth numbers */}
      <div className="session-strip">
        <span className="ss-item font-mono ss-steps">
          Steps: <strong><SmoothNumber value={session.summary.total_steps_executed} /></strong>
        </span>
        <span className="ss-sep" />
        <span className="ss-item font-mono ss-anom">
          Anomalies: <strong><SmoothNumber value={session.summary.anomalies_flagged} /></strong>
        </span>
        <span className="ss-sep" />
        <span className="ss-item font-mono ss-elapsed">
          Elapsed: <strong><SmoothNumber value={session.elapsed_seconds} precision={0} suffix="s" /></strong>
        </span>
        <span className="ss-sep" />
        <span className="ss-item font-mono ss-rec">
          Recorded: <strong><SmoothNumber value={session.total_actions_recorded} /></strong>
        </span>
      </div>

      <div style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <div
          className="logs-scroll"
          ref={scrollRef}
          onScroll={handleScroll}
        >
          {logs.map(entry => (
            <div key={entry.id} className={`log-row log-${entry.level.toLowerCase()}`}>
              <span className="log-ts font-mono">{entry.timestamp}</span>
              <span className={`log-level font-mono log-level-${entry.level.toLowerCase()}`}>
                {entry.level.padEnd(7)}
              </span>
              <span className="log-msg">{entry.message}</span>
            </div>
          ))}
        </div>

        {!autoScroll && (
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

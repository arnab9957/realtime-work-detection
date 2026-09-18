import { useEffect, useRef, useState } from 'react';
import type { TelemetryData } from '../../types/api';

interface AnomalyDetectionProps {
  telemetry: TelemetryData;
}

export function AnomalyDetection({ telemetry }: AnomalyDetectionProps) {
  const anomalyActive = telemetry.anomaly && telemetry.anomaly !== 'NONE';
  const [lastAnomaly, setLastAnomaly] = useState<string>('NONE');
  const [eventTime, setEventTime] = useState<string>('');
  const [speaking, setSpeaking] = useState(false);
  const synthRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Trigger TTS only on new anomaly state transition
  useEffect(() => {
    const current = telemetry.anomaly;
    if (current === lastAnomaly) return;

    setLastAnomaly(current);

    if (current && current !== 'NONE') {
      setEventTime(nowHHMMSS());
      speakAnomaly(current, setSpeaking, synthRef);
    }
  }, [telemetry.anomaly, lastAnomaly]);

  return (
    <div className={`panel anomaly-panel ${anomalyActive ? 'anomaly-active' : ''}`}>
      <div className="panel-header">
        <span className="panel-title">ANOMALY DETECTION</span>
        {anomalyActive && (
          <span className="anomaly-severity-badge">WARNING</span>
        )}
      </div>

      <div className="anomaly-body">
        {!anomalyActive ? (
          <div className="anomaly-normal">
            <span className="status-dot status-dot-green" />
            <span className="anomaly-normal-text">NORMAL</span>
          </div>
        ) : (
          <div className="anomaly-alert">
            <div className="anomaly-icon-row">
              <span className="anomaly-warn-icon">⚠</span>
              <span className="anomaly-warn-label">ANOMALY DETECTED</span>
            </div>
            <div className="anomaly-code">{telemetry.anomaly}</div>
          </div>
        )}

        <div className="anomaly-details">
          <AnomalyRow label="Step Verdict" value={telemetry.step_verdict || '—'} />
          <AnomalyRow
            label="Step Status"
            value={telemetry.is_step_correct ? 'CORRECT' : 'DEVIATION'}
            highlight={!telemetry.is_step_correct}
          />
          <AnomalyRow
            label="Latest Event"
            value={anomalyActive ? telemetry.anomaly : 'No anomaly detected'}
          />
          {eventTime && anomalyActive && (
            <AnomalyRow label="Detected At" value={eventTime} mono />
          )}
        </div>

        {/* TTS status */}
        {speaking && (
          <div className="tts-indicator">
            <WaveformIcon />
            <span className="tts-label">SPEAKING...</span>
          </div>
        )}
      </div>
    </div>
  );
}

function AnomalyRow({ label, value, highlight, mono }: {
  label: string;
  value: string;
  highlight?: boolean;
  mono?: boolean;
}) {
  return (
    <div className="anomaly-row">
      <span className="anomaly-row-label">{label}</span>
      <span className={`anomaly-row-value ${highlight ? 'anomaly-row-warn' : ''} ${mono ? 'font-mono' : ''}`}>
        {value}
      </span>
    </div>
  );
}

function WaveformIcon() {
  return (
    <svg className="waveform-icon" viewBox="0 0 40 20" fill="none">
      <rect x="0"  y="8"  width="4" height="4"  rx="1" fill="#f59e0b" className="wave-bar b1" />
      <rect x="6"  y="4"  width="4" height="12" rx="1" fill="#f59e0b" className="wave-bar b2" />
      <rect x="12" y="1"  width="4" height="18" rx="1" fill="#f59e0b" className="wave-bar b3" />
      <rect x="18" y="5"  width="4" height="10" rx="1" fill="#f59e0b" className="wave-bar b4" />
      <rect x="24" y="8"  width="4" height="4"  rx="1" fill="#f59e0b" className="wave-bar b5" />
      <rect x="30" y="3"  width="4" height="14" rx="1" fill="#f59e0b" className="wave-bar b6" />
      <rect x="36" y="6"  width="4" height="8"  rx="1" fill="#f59e0b" className="wave-bar b7" />
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

function nowHHMMSS(): string {
  return new Date().toTimeString().slice(0, 8);
}

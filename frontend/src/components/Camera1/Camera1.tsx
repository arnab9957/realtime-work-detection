import { useEffect, useRef, useState } from 'react';
import { Layers, ArrowRight, CheckCircle2, AlertTriangle } from 'lucide-react';
import { STEP_LABELS } from '../../mock/fallbackData';
import { API } from '../../services/api';
import type { TelemetryData } from '../../types/api';

interface Camera1Props {
  telemetry: TelemetryData;
}

export function Camera1({ telemetry }: Camera1Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const fallbackRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [streamBroken, setStreamBroken] = useState(false);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    const onError = () => {
      setStreamBroken(true);
      if (fallbackRef.current) return;
      fallbackRef.current = setInterval(() => {
        if (img) img.src = `${API.SNAPSHOT}?t=${Date.now()}`;
      }, 200);
    };

    const onLoad = () => {
      if (img.src.includes('/stream')) {
        setStreamBroken(false);
        if (fallbackRef.current) {
          clearInterval(fallbackRef.current);
          fallbackRef.current = null;
        }
      }
    };

    img.addEventListener('error', onError);
    img.addEventListener('load', onLoad);
    return () => {
      img.removeEventListener('error', onError);
      img.removeEventListener('load', onLoad);
      if (fallbackRef.current) clearInterval(fallbackRef.current);
    };
  }, []);

  const anomalyActive = telemetry.anomaly && telemetry.anomaly !== 'NONE';
  const prevStepLabel = resolvePrevStepLabel(telemetry.step_name, telemetry.step);

  return (
    <div className="panel camera1-panel">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Layers size={14} strokeWidth={2} style={{ color: 'var(--cyan)' }} />
          <span className="panel-title">Camera 01 — HAR Output</span>
        </div>
        <span className={`feed-badge ${streamBroken ? 'badge-warn' : 'badge-live'}`}>
          {streamBroken ? 'Snapshot Mode' : 'Live Stream'}
        </span>
      </div>

      {/* ── Stream area ─────────────────────────────────────── */}
      <div className="stream-wrapper">
        <img
          ref={imgRef}
          src={API.STREAM}
          alt="Live HAR stream"
          className="stream-img"
        />

        {/* Frame / FPS telemetry */}
        <div className="stream-overlay-top">
          <span className="overlay-chip">FRM {String(telemetry.frame_id).padStart(6, '0')}</span>
          <span className="overlay-chip">{telemetry.fps > 0 ? `${telemetry.fps.toFixed(1)} fps` : '— fps'}</span>
          <span className="overlay-chip">{telemetry.latency_ms > 0 ? `${telemetry.latency_ms.toFixed(0)} ms` : ''}</span>
        </div>

        {/* Anomaly banner */}
        {anomalyActive && (
          <div className="anomaly-banner">
            <AlertTriangle size={13} strokeWidth={2} />
            <span>{telemetry.anomaly}</span>
          </div>
        )}

        {/* Step verdict */}
        <div className={`step-verdict ${telemetry.is_step_correct ? 'verdict-ok' : 'verdict-err'}`}>
          {telemetry.is_step_correct
            ? <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle2 size={11} strokeWidth={2} /> {telemetry.step_verdict || 'Nominal'}</span>
            : <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><AlertTriangle size={11} strokeWidth={2} /> {telemetry.step_verdict || 'Deviation'}</span>
          }
        </div>
      </div>

      {/* ── Information strip ───────────────────────────────── */}
      <div className="cam1-info-strip">
        <InfoBlock
          label="Previous Step"
          value={prevStepLabel}
          variant="dim"
        />
        <div className="info-sep" />
        <InfoBlock
          label="Current Action"
          value={
            telemetry.what_i_am_doing && telemetry.what_i_am_doing !== 'IDLE'
              ? telemetry.what_i_am_doing
              : 'Observing workspace'
          }
          variant="highlight"
        />
        <div className="info-sep" />
        <InfoBlock
          label="Expected Next"
          value={telemetry.what_i_have_to_do || '—'}
          icon={<ArrowRight size={10} strokeWidth={2} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: 2 }} />}
        />
        <div className="info-sep" />
        <InfoBlock
          label="Step Status"
          value={telemetry.step_verdict || 'Awaiting'}
          variant={anomalyActive ? 'warn' : telemetry.is_step_correct ? 'ok' : 'err'}
        />
      </div>
    </div>
  );
}

function InfoBlock({
  label,
  value,
  variant,
  icon,
}: {
  label: string;
  value: string;
  variant?: 'dim' | 'highlight' | 'ok' | 'warn' | 'err';
  icon?: React.ReactNode;
}) {
  const valueClass = [
    'info-value',
    variant === 'dim'       ? 'info-dim'       : '',
    variant === 'highlight' ? 'info-highlight'  : '',
    variant === 'ok'        ? 'info-ok'         : '',
    variant === 'warn'      ? 'info-warn'        : '',
    variant === 'err'       ? 'info-err'         : '',
  ].filter(Boolean).join(' ');

  return (
    <div className="info-block">
      <span className="info-label">{label}</span>
      <span className={valueClass} style={icon ? { display: 'flex', alignItems: 'flex-start', gap: 4 } : undefined}>
        {icon}
        {value}
      </span>
    </div>
  );
}

function resolvePrevStepLabel(currentStepName: string, stepIdx: number): string {
  const keys = Object.keys(STEP_LABELS);
  const currentIdx = keys.indexOf(currentStepName);
  if (currentIdx > 0) {
    const prevKey = keys[currentIdx - 1];
    return STEP_LABELS[prevKey] || `Step ${stepIdx - 1}`;
  }
  if (stepIdx > 0) return `Step ${stepIdx - 1}`;
  return '—';
}

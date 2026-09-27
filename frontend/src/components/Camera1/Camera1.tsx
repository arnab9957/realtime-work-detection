import { useEffect, useRef, useState } from 'react';
import { Layers, ArrowRight, CheckCircle2, AlertTriangle } from 'lucide-react';
import { STEP_LABELS, RED_YELLOW_STEPS } from '../../mock/fallbackData';
import { API } from '../../services/api';
import type { TelemetryData } from '../../types/api';
import { SmoothNumber, AnimatedText } from '../common/SmoothData';

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
        // Flicker-free double buffering for snapshot mode:
        // Preload image off-screen and swap src only once loaded into GPU buffer
        const preloader = new Image();
        preloader.onload = () => {
          if (imgRef.current) imgRef.current.src = preloader.src;
        };
        preloader.src = `${API.SNAPSHOT}?t=${Date.now()}`;
      }, 250);
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

        {/* Frame / FPS telemetry with tabular nums & smooth interpolation */}
        <div className="stream-overlay-top">
          <span className="overlay-chip">
            <SmoothNumber value={telemetry.frame_id} padLength={6} prefix="FRM " />
          </span>
          <span className="overlay-chip">
            <SmoothNumber
              value={telemetry.fps > 0 ? telemetry.fps : null}
              precision={1}
              suffix=" fps"
              fallback="— fps"
              flash
            />
          </span>
          {telemetry.latency_ms > 0 && (
            <span className="overlay-chip">
              <SmoothNumber
                value={telemetry.latency_ms}
                precision={0}
                suffix=" ms"
                flash
              />
            </span>
          )}
        </div>

        {/* Anomaly banner */}
        {anomalyActive && (
          <div className="anomaly-banner">
            <AlertTriangle size={13} strokeWidth={2} />
            <AnimatedText inline>{telemetry.anomaly}</AnimatedText>
          </div>
        )}

        {/* Step verdict */}
        <div className={`step-verdict ${telemetry.is_step_correct ? 'verdict-ok' : 'verdict-err'}`}>
          {telemetry.is_step_correct ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <CheckCircle2 size={11} strokeWidth={2} />
              <AnimatedText inline>{telemetry.step_verdict || 'Nominal'}</AnimatedText>
            </span>
          ) : (
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <AlertTriangle size={11} strokeWidth={2} />
              <AnimatedText inline>{telemetry.step_verdict || 'Deviation'}</AnimatedText>
            </span>
          )}
        </div>
      </div>

      {/* ── Information strip ───────────────────────────────── */}
      <div className="cam1-info-strip">
        <InfoBlock
          label="Previous Step"
          value={<AnimatedText>{prevStepLabel}</AnimatedText>}
          variant="dim"
        />
        <div className="info-sep" />
        <InfoBlock
          label="Current Action"
          value={
            <AnimatedText>
              {telemetry.what_i_am_doing && telemetry.what_i_am_doing !== 'IDLE'
                ? telemetry.what_i_am_doing
                : 'Observing workspace'}
            </AnimatedText>
          }
          variant="highlight"
        />
        <div className="info-sep" />
        <InfoBlock
          label="Expected Next"
          value={<AnimatedText>{telemetry.what_i_have_to_do || '—'}</AnimatedText>}
          icon={<ArrowRight size={10} strokeWidth={2} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: 2 }} />}
        />
        <div className="info-sep" />
        <InfoBlock
          label="Step Status"
          value={<AnimatedText>{telemetry.step_verdict || 'Awaiting'}</AnimatedText>}
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
  value: React.ReactNode;
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

function resolvePrevStepLabel(_currentStepName: string, stepIdx: number): string {
  if (stepIdx <= 0) return '—';
  const prevStepIdx = stepIdx - 1;
  const steps = RED_YELLOW_STEPS;
  if (prevStepIdx >= 0 && prevStepIdx < steps.length) {
    const prevKey = steps[prevStepIdx].key;
    return STEP_LABELS[prevKey] || steps[prevStepIdx].label || `Step ${prevStepIdx}`;
  }
  return `Step ${prevStepIdx}`;
}

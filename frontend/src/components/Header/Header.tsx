import { useEffect, useState } from 'react';
import {
  Activity, Cpu, Gauge, MemoryStick, MonitorPlay,
  ChevronDown, RotateCcw, Camera, Video, FlaskConical,
  Radio,
} from 'lucide-react';
import { getSimulatedSystemMetrics } from '../../mock/fallbackData';
import { API } from '../../services/api';
import type { TelemetryData } from '../../types/api';

interface HeaderProps {
  telemetry: TelemetryData;
  connected: boolean;
}

export function Header({ telemetry, connected }: HeaderProps) {
  const [sysMetrics, setSysMetrics] = useState(getSimulatedSystemMetrics());
  const [controlsOpen, setControlsOpen] = useState(false);

  // Slowly drift simulated system metrics
  useEffect(() => {
    const id = setInterval(() => {
      setSysMetrics(prev => ({
        cpu:  clamp(prev.cpu  + (Math.random() * 4 - 2), 15, 80),
        gpu:  clamp(prev.gpu  + (Math.random() * 6 - 3), 30, 95),
        ram:  clamp(prev.ram  + (Math.random() * 2 - 1), 30, 80),
        vram: clamp(prev.vram + (Math.random() * 4 - 2), 30, 90),
      }));
    }, 2500);
    return () => clearInterval(id);
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!controlsOpen) return;
    const onOutside = () => setControlsOpen(false);
    setTimeout(() => document.addEventListener('click', onOutside), 0);
    return () => document.removeEventListener('click', onOutside);
  }, [controlsOpen]);

  const sourceLabel =
    telemetry.source_type === 'LIVE_WEBCAM' ? 'LIVE CAM' :
    telemetry.source_type?.includes('red_yellow') ? 'RED-YELLOW' :
    telemetry.source_type || '—';

  async function handleReset() {
    try { await fetch(API.RESET); } catch {}
    setControlsOpen(false);
  }
  async function handleSource(src: string) {
    try { await fetch(API.SOURCE(src)); } catch {}
    setControlsOpen(false);
  }
  async function handleExperiment(exp: string) {
    try { await fetch(API.EXPERIMENT(exp)); } catch {}
    setControlsOpen(false);
  }

  return (
    <header className="header">
      {/* ── Brand + experiment info ──────────────────── */}
      <div className="header-left">
        <div className="header-brand">
          <span className="brand-dot" />
          <span className="brand-title">BAS — HAR System</span>
        </div>

        <div className="header-meta">
          <MetricChip icon={<FlaskConical size={11} strokeWidth={1.8} />} label="EXP" value={telemetry.experiment_id || '—'} />
          <MetricChip icon={<MonitorPlay size={11} strokeWidth={1.8} />} label="SRC" value={sourceLabel} />
        </div>
      </div>

      {/* ── Center: hardware readouts ─────────────────── */}
      <div className="header-metrics">
        <MetricReadout icon={<Cpu size={12} strokeWidth={1.5} />} label="CPU" value={`${Math.round(sysMetrics.cpu)}%`} note="est." />
        <div className="metric-sep" />
        <MetricReadout icon={<Activity size={12} strokeWidth={1.5} />} label="GPU" value={`${Math.round(sysMetrics.gpu)}%`} note="est." />
        <div className="metric-sep" />
        <MetricReadout icon={<MemoryStick size={12} strokeWidth={1.5} />} label="RAM" value={`${Math.round(sysMetrics.ram)}%`} note="est." />
        <div className="metric-sep" />
        <MetricReadout icon={<Gauge size={12} strokeWidth={1.5} />} label="VRAM" value={`${Math.round(sysMetrics.vram)}%`} note="est." />
        <div className="metric-sep" />
        <MetricReadout icon={<Video size={12} strokeWidth={1.5} />} label="FPS" value={telemetry.fps > 0 ? telemetry.fps.toFixed(1) : '—'} />
        <div className="metric-sep" />
        <MetricReadout icon={<Radio size={12} strokeWidth={1.5} />} label="LAT" value={telemetry.latency_ms > 0 ? `${telemetry.latency_ms.toFixed(0)} ms` : '—'} />
      </div>

      {/* ── Right: status + controls ─────────────────── */}
      <div className="header-right">
        <div className={`status-pill ${connected ? 'status-online' : 'status-offline'}`}>
          <span className="status-dot" />
          <span>{connected ? 'SYSTEM ONLINE' : 'OFFLINE'}</span>
        </div>

        <div className="controls-wrapper" onClick={e => e.stopPropagation()}>
          <button
            className="ctrl-btn"
            onClick={() => setControlsOpen(o => !o)}
            aria-expanded={controlsOpen}
          >
            <Camera size={13} strokeWidth={1.8} />
            <span>Controls</span>
            <ChevronDown size={12} strokeWidth={2} style={{ opacity: 0.6, transition: 'transform 0.15s', transform: controlsOpen ? 'rotate(180deg)' : 'rotate(0deg)' }} />
          </button>

          {controlsOpen && (
            <div className="ctrl-dropdown">
              <p className="ctrl-group-label">System</p>
              <button className="ctrl-item" onClick={handleReset}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <RotateCcw size={11} strokeWidth={2} style={{ color: 'var(--text-muted)' }} />
                  Reset FSM
                </span>
              </button>

              <p className="ctrl-group-label">Video Source</p>
              <button className="ctrl-item" onClick={() => handleSource('cam')}>Live Camera</button>
              <button className="ctrl-item" onClick={() => handleSource('red_yellow.mp4')}>Red-Yellow Clip</button>
              <button className="ctrl-item" onClick={() => handleSource('clip1.mp4')}>Demo Clip</button>

              <p className="ctrl-group-label">Experiment</p>
              <button className="ctrl-item" onClick={() => handleExperiment('isro_dual')}>ISRO Dual (26174)</button>
              <button className="ctrl-item" onClick={() => handleExperiment('box_return')}>Box Return</button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

function MetricReadout({ icon, label, value, note }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="metric-item">
      <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
        <span style={{ color: 'var(--text-muted)', display: 'flex' }}>{icon}</span>
        {label}
      </span>
      <span className="metric-value">{value}</span>
      {note && <span className="metric-note">{note}</span>}
    </div>
  );
}

function MetricChip({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric-chip">
      <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center' }}>{icon}</span>
      <span className="metric-chip-label">{label}</span>
      <span className="metric-chip-value">{value}</span>
    </div>
  );
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

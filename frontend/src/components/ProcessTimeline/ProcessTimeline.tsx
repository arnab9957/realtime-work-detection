import { useEffect, useRef } from 'react';
import { Activity } from 'lucide-react';
import { PROCESS_STEPS, STEP_LABELS } from '../../mock/fallbackData';
import type { TelemetryData } from '../../types/api';

interface ProcessTimelineProps {
  telemetry: TelemetryData;
}

export function ProcessTimeline({ telemetry }: ProcessTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Map step_name to a PROCESS_STEPS index
  const activeIdx = resolveActiveStep(telemetry.step_name, telemetry.step);
  const isComplete =
    telemetry.step_name === 'COMPLETE' ||
    telemetry.step_name === 'BOX_CLOSED' ||
    telemetry.step_name === 'DUAL_COMPLETE';

  // Auto-scroll active step into view
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const activeEl = container.querySelector('.timeline-step.tl-active') as HTMLElement | null;
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [activeIdx]);

  return (
    <div className="panel timeline-panel">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Activity size={14} strokeWidth={2} style={{ color: 'var(--green)' }} />
          <span className="panel-title">PROCESS TIMELINE</span>
        </div>
        <span className="panel-subtitle font-mono">
          STEP {telemetry.step} · {STEP_LABELS[telemetry.step_name] || telemetry.step_name}
        </span>
      </div>

      <div className="timeline-scroll-wrapper" ref={containerRef}>
        <div className="timeline-track">
          {PROCESS_STEPS.map((step, idx) => {
            const completed = isComplete || idx < activeIdx;
            const active = !isComplete && idx === activeIdx;
            const upcoming = !isComplete && idx > activeIdx;

            return (
              <div
                key={step.key}
                className={[
                  'timeline-step',
                  completed ? 'tl-completed' : '',
                  active ? 'tl-active' : '',
                  upcoming ? 'tl-upcoming' : '',
                ].filter(Boolean).join(' ')}
              >
                {/* Connector line before step (not before first) */}
                {idx > 0 && (
                  <div className={`tl-connector ${completed || active ? 'tl-conn-done' : ''}`} />
                )}

                {/* Circle badge */}
                <div className="tl-circle">
                  {completed ? (
                    <svg viewBox="0 0 16 16" fill="none" className="tl-check">
                      <polyline points="3 8 7 12 13 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                  ) : (
                    <span className="tl-num">{idx + 1}</span>
                  )}
                  {active && <span className="tl-pulse" />}
                </div>

                {/* Label */}
                <span className="tl-label">{step.shortLabel}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** Map backend step_name → PROCESS_STEPS index */
function resolveActiveStep(stepName: string, stepIdx: number): number {
  // Direct key match in PROCESS_STEPS
  const directIdx = PROCESS_STEPS.findIndex(s => s.key === stepName);
  if (directIdx >= 0) return directIdx;
  // Fallback to FSM step integer clamped to array bounds
  return Math.min(Math.max(stepIdx, 0), PROCESS_STEPS.length - 1);
}

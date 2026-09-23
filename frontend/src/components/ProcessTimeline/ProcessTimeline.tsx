import { useEffect, useRef } from 'react';
import { Activity } from 'lucide-react';
import { RED_YELLOW_STEPS, BOX_RETURN_STEPS, STEP_LABELS, type ProcessStep } from '../../mock/fallbackData';
import type { TelemetryData } from '../../types/api';

interface ProcessTimelineProps {
  telemetry: TelemetryData;
}

export function ProcessTimeline({ telemetry }: ProcessTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Select procedure steps based on active experiment protocol
  const steps: ProcessStep[] =
    telemetry.experiment_id && telemetry.experiment_id.includes('BOX-RETURN')
      ? BOX_RETURN_STEPS
      : RED_YELLOW_STEPS;

  // Map step integer & step_name to the correct process steps index
  const activeIdx = resolveActiveStep(steps, telemetry.step_name, telemetry.step);
  const maxStepIdx = steps.length - 1;
  const isComplete =
    telemetry.step >= maxStepIdx ||
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
          {steps.map((step, idx) => {
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

/** Map backend step integer & step_name → steps index */
function resolveActiveStep(steps: ProcessStep[], stepName: string, stepIdx: number): number {
  // If valid integer step provided by FSM (0 .. steps.length - 1), it is the canonical ground truth
  if (typeof stepIdx === 'number' && stepIdx >= 0 && stepIdx < steps.length) {
    return stepIdx;
  }
  // Fallback to key or alias match
  const directIdx = steps.findIndex(
    s => s.key === stepName || (s.aliases && s.aliases.includes(stepName))
  );
  if (directIdx >= 0) return directIdx;
  return Math.min(Math.max(stepIdx || 0, 0), steps.length - 1);
}

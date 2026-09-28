import { useEffect, useRef, useState, useMemo } from 'react';
import { Activity, Play, RotateCcw, CheckCircle2, Pause } from 'lucide-react';
import { RED_YELLOW_STEPS, BOX_RETURN_STEPS, STEP_LABELS, type ProcessStep } from '../../mock/fallbackData';
import type { TelemetryData } from '../../types/api';
import { API } from '../../services/api';
import { AnimatedText, SmoothNumber } from '../common/SmoothData';
import { HumanConsentModal, type ConsentActionType, type FeedMode } from '../common/HumanConsentModal';

interface ProcessTimelineProps {
  telemetry: TelemetryData;
}

export function ProcessTimeline({ telemetry }: ProcessTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [modalAction, setModalAction] = useState<ConsentActionType | null>(null);
  const [isStopped, setIsStopped] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);
  const [currentFeedMode, setCurrentFeedMode] = useState<FeedMode>('dummy');

  // Sync feed mode if telemetry reveals active source
  useEffect(() => {
    if (telemetry.source_type) {
      if (telemetry.source_type.includes('CAM') || telemetry.source_type.includes('LIVE') || telemetry.source_type === '0') {
        setCurrentFeedMode('live');
      } else if (telemetry.source_type.includes('CLIP') || telemetry.source_type.includes('c1')) {
        setCurrentFeedMode('dummy');
      }
    }
  }, [telemetry.source_type]);

  // Select procedure steps based on active experiment protocol
  const steps: ProcessStep[] = useMemo(() => {
    return telemetry.experiment_id && telemetry.experiment_id.includes('BOX-RETURN')
      ? BOX_RETURN_STEPS
      : RED_YELLOW_STEPS;
  }, [telemetry.experiment_id]);

  // Map step integer & step_name to the correct process steps index
  const activeIdx = resolveActiveStep(steps, telemetry.step_name, telemetry.step);
  const maxStepIdx = steps.length - 1;

  // Process completion detection
  const isComplete =
    telemetry.step >= maxStepIdx ||
    telemetry.step_name === 'COMPLETE' ||
    telemetry.step_name === 'BOX_CLOSED' ||
    telemetry.step_name === 'DUAL_COMPLETE';

  // Calculate percentage: locked at 100% when complete, otherwise calculated from activeIdx
  const progressPercent = useMemo(() => {
    if (isComplete) return 100;
    if (maxStepIdx <= 0) return 0;
    return Math.min(100, Math.max(0, Math.round((activeIdx / maxStepIdx) * 100)));
  }, [isComplete, activeIdx, maxStepIdx]);

  // Auto-scroll active step into view with smooth behavior (only when running)
  useEffect(() => {
    if (isComplete || isStopped) return;
    const container = containerRef.current;
    if (!container) return;
    const activeEl = container.querySelector('.timeline-step.tl-active') as HTMLElement | null;
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [activeIdx, isComplete, isStopped]);

  const stepTitle = STEP_LABELS[telemetry.step_name] || telemetry.step_name || 'INITIALIZING';

  // Human Consent action executor: switches source (c1.mp4 vs Live camera) and executes action
  const handleConfirmConsent = async (action: ConsentActionType, feedMode: FeedMode) => {
    setCurrentFeedMode(feedMode);
    try {
      // 1. Switch the video source on the streaming server
      const targetSource = feedMode === 'dummy' ? 'c1.mp4' : '0';
      await fetch(API.SOURCE(targetSource));

      // 2. Trigger Reset or Start
      if (action === 'RESET') {
        await fetch(`${API.RESET}?t=${Date.now()}`);
        setIsStopped(false);
        setFeedbackMsg(
          feedMode === 'dummy'
            ? 'Human Consent: Resetting test sequence using Dummy feed (c1.mp4).'
            : 'Human Consent: Resetting test sequence using Live Camera.'
        );
      } else if (action === 'START') {
        if (isComplete) {
          // Restarting complete sequence: reset then start
          await fetch(`${API.RESET}?t=${Date.now()}`);
        }
        await fetch(`${API.START}?t=${Date.now()}`);
        setIsStopped(false);
        setFeedbackMsg(
          feedMode === 'dummy'
            ? 'Human Consent: Started processing c1.mp4 video sequence.'
            : 'Human Consent: Live Camera sequence initiated.'
        );
      }

      // 3. Dispatch stream refresh for Camera components
      window.dispatchEvent(
        new CustomEvent('bas-source-switch', { detail: { mode: feedMode, source: targetSource } })
      );
    } catch (err) {
      console.warn('Backend endpoint unreachable, running in local mode:', err);
      setIsStopped(false);
      setFeedbackMsg(
        `Human Consent: ${action} dispatched (${feedMode === 'dummy' ? 'c1.mp4' : 'Live Camera'}).`
      );
      window.dispatchEvent(new CustomEvent('bas-source-switch', { detail: { mode: feedMode } }));
    }

    // Auto-clear feedback after 4 seconds
    setTimeout(() => {
      setFeedbackMsg(null);
    }, 4000);
  };

  return (
    <div className="panel timeline-panel">
      {/* ── Panel Header ──────────────────────────────────────── */}
      <div className="panel-header tl-panel-header">
        {/* Left: Title + Real-time Status */}
        <div className="tl-header-left">
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <Activity size={14} strokeWidth={2} style={{ color: 'var(--green)' }} />
            <span className="panel-title">PROCESS TIMELINE</span>
          </div>

          {/* Status Indicator */}
          {isComplete ? (
            <span className="tl-status-badge tl-badge-complete">
              <CheckCircle2 size={12} strokeWidth={2.5} />
              <span>COMPLETED (STOPPED)</span>
            </span>
          ) : isStopped ? (
            <span className="tl-status-badge tl-badge-stopped">
              <Pause size={12} strokeWidth={2.5} />
              <span>STOPPED</span>
            </span>
          ) : (
            <span className="tl-live-indicator">
              <span className="tl-live-dot" />
              <span className="tl-live-text">
                STEP <SmoothNumber value={telemetry.step} precision={0} />
              </span>
              <span className="tl-live-tag">IN PROGRESS</span>
            </span>
          )}
          {/* Active Feed Source Badge */}
          <span className="tl-source-badge font-mono" title="Active Experiment Feed Source">
            {currentFeedMode === 'dummy' ? 'DUMMY (c1.mp4)' : 'LIVE CAMERA'}
          </span>
        </div>

        {/* Right: Progress % + Subtitle + Action Controls */}
        <div className="timeline-header-meta">
          {/* Progress Percentage Badge */}
          <div className="tl-pct-badge" title="Procedure Completion Percentage">
            <span className="tl-pct-value">{progressPercent}%</span>
            <span className="tl-pct-label">
              {isComplete ? 'DONE' : isStopped ? 'HALTED' : 'PROGRESS'}
            </span>
          </div>

          {/* Step description */}
          <span className="panel-subtitle font-mono tl-step-subtitle">
            <AnimatedText inline>{stepTitle}</AnimatedText>
          </span>

          {/* ── Action Buttons: Reset & Start ── */}
          <div className="tl-action-group">
            <button
              type="button"
              className="tl-action-btn tl-btn-reset"
              onClick={() => setModalAction('RESET')}
              title="Reset procedure with Human Consent verification"
            >
              <RotateCcw size={12} strokeWidth={2.2} />
              <span>Reset</span>
            </button>

            <button
              type="button"
              className={`tl-action-btn ${isComplete ? 'tl-btn-restart' : 'tl-btn-start'}`}
              onClick={() => setModalAction('START')}
              title={
                isComplete
                  ? 'Restart procedure with Human Consent verification'
                  : 'Start procedure with Human Consent verification'
              }
            >
              <Play size={12} strokeWidth={2.2} fill="currentColor" />
              <span>{isComplete ? 'Restart' : 'Start'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Linear Progress Bar ───────────────────────────────── */}
      <div className="tl-linear-progress-wrapper" role="progressbar" aria-valuenow={progressPercent} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={[
            'tl-linear-progress-bar',
            isComplete ? 'progress-completed' : '',
            isStopped ? 'progress-stopped' : '',
            !isComplete && !isStopped ? 'progress-running' : '',
          ].filter(Boolean).join(' ')}
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Toast Feedback for Consent Confirmation */}
      {feedbackMsg && (
        <div className="tl-feedback-toast">
          <CheckCircle2 size={13} className="text-green" />
          <span>{feedbackMsg}</span>
        </div>
      )}

      {/* ── Step-by-Step Milestones Track ────────────────────── */}
      <div className="timeline-scroll-wrapper" ref={containerRef}>
        <div className="timeline-track">
          {steps.map((step, idx) => {
            const completed = isComplete || idx < activeIdx;
            const active = !isComplete && idx === activeIdx;
            const upcoming = !isComplete && idx > activeIdx;
            const isLeadingToActive = !isComplete && idx === activeIdx;

            return (
              <div
                key={step.key}
                className={[
                  'timeline-step',
                  completed ? 'tl-completed' : '',
                  active ? 'tl-active' : '',
                  upcoming ? 'tl-upcoming' : '',
                  isComplete ? 'tl-done-frozen' : '',
                ].filter(Boolean).join(' ')}
              >
                {/* Connector line before step (not before first) */}
                {idx > 0 && (
                  <div
                    className={[
                      'tl-connector',
                      completed ? 'tl-conn-done' : '',
                      isLeadingToActive ? 'tl-conn-active tl-conn-done' : '',
                      isComplete || isStopped ? 'tl-conn-stopped' : '',
                    ].filter(Boolean).join(' ')}
                  />
                )}

                {/* Circle badge with state icons */}
                <div className="tl-circle">
                  {completed ? (
                    <svg viewBox="0 0 16 16" fill="none" className="tl-check">
                      <polyline
                        points="3 8 7 12 13 4"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  ) : (
                    <span className="tl-num">{idx + 1}</span>
                  )}
                  {/* Pulse rings: only animate when active and NOT complete */}
                  {active && !isComplete && !isStopped && (
                    <>
                      <span className="tl-pulse" />
                      <span className="tl-pulse-2" />
                    </>
                  )}
                </div>

                {/* Short Step Label */}
                <span className="tl-label">{step.shortLabel}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Completion Banner (Shown when process completes) ──── */}
      {isComplete && (
        <div className="tl-completion-bar">
          <div className="tl-completion-bar-left">
            <CheckCircle2 size={15} className="tl-completion-icon" />
            <div>
              <span className="tl-completion-text">
                Procedure Sequence Completed — Progress in Progress Stopped.
              </span>
              <span className="tl-completion-sub">
                Autonomous sequence validation finished nominal. Ready for Human Consent authorization.
              </span>
            </div>
          </div>
          <div className="tl-completion-bar-actions">
            <button
              type="button"
              className="tl-action-btn tl-btn-reset-sm"
              onClick={() => setModalAction('RESET')}
            >
              <RotateCcw size={11} />
              <span>Reset Sequence</span>
            </button>
            <button
              type="button"
              className="tl-action-btn tl-btn-restart-sm"
              onClick={() => setModalAction('START')}
            >
              <Play size={11} fill="currentColor" />
              <span>Restart Run</span>
            </button>
          </div>
        </div>
      )}

      {/* ── Human Consent Modal with Blur Background ─────────── */}
      <HumanConsentModal
        isOpen={modalAction !== null}
        actionType={modalAction || 'START'}
        experimentId={telemetry.experiment_id || 'BAS-EXP-26174'}
        currentStep={telemetry.step}
        currentStepName={stepTitle}
        isComplete={isComplete}
        initialFeedMode={currentFeedMode}
        onClose={() => setModalAction(null)}
        onConfirm={handleConfirmConsent}
      />
    </div>
  );
}

/** Map backend step integer & step_name → steps index */
function resolveActiveStep(steps: ProcessStep[], stepName: string, stepIdx: number): number {
  if (typeof stepIdx === 'number' && stepIdx >= 0 && stepIdx < steps.length) {
    return stepIdx;
  }
  const directIdx = steps.findIndex(
    s => s.key === stepName || (s.aliases && s.aliases.includes(stepName))
  );
  if (directIdx >= 0) return directIdx;
  return Math.min(Math.max(stepIdx || 0, 0), steps.length - 1);
}

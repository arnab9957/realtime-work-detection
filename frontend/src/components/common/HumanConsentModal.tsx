import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  ShieldAlert,
  ShieldCheck,
  RotateCcw,
  Play,
  X,
  Fingerprint,
  AlertTriangle,
  SlidersHorizontal,
  Film,
  Camera,
} from 'lucide-react';

export type ConsentActionType = 'START' | 'RESET';
export type FeedMode = 'dummy' | 'live';

interface HumanConsentModalProps {
  isOpen: boolean;
  actionType: ConsentActionType;
  experimentId: string;
  currentStep: number;
  currentStepName: string;
  isComplete: boolean;
  initialFeedMode?: FeedMode;
  onClose: () => void;
  onConfirm: (action: ConsentActionType, feedMode: FeedMode) => Promise<void> | void;
}

export function HumanConsentModal({
  isOpen,
  actionType,
  experimentId,
  currentStep,
  currentStepName,
  isComplete,
  initialFeedMode = 'dummy',
  onClose,
  onConfirm,
}: HumanConsentModalProps) {
  const [feedMode, setFeedMode] = useState<FeedMode>(initialFeedMode);
  const [confirmedConsent, setConfirmedConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Lock body scroll while modal is active
  useEffect(() => {
    if (isOpen) {
      const prevOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = prevOverflow;
      };
    }
  }, [isOpen]);

  // Reset consent checkbox and mode whenever modal opens or actionType changes
  useEffect(() => {
    if (isOpen) {
      setConfirmedConsent(false);
      setSubmitting(false);
      setFeedMode(initialFeedMode || 'dummy');
    }
  }, [isOpen, actionType, initialFeedMode]);

  // Handle ESC key to dismiss
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !submitting) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, submitting, onClose]);

  if (!isOpen || typeof document === 'undefined') return null;

  const isReset = actionType === 'RESET';

  const handleAuthorize = async () => {
    if (!confirmedConsent || submitting) return;
    setSubmitting(true);
    try {
      await onConfirm(actionType, feedMode);
      onClose();
    } catch (err) {
      console.error('Human consent action failed:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return createPortal(
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget && !submitting) {
          onClose();
        }
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="consent-modal-title"
    >
      <div className="modal-dialog">
        {/* Top Header */}
        <div className="modal-header">
          <div className="modal-header-left">
            <div className={`modal-icon-badge ${isReset ? 'badge-amber' : 'badge-green'}`}>
              {isReset ? (
                <ShieldAlert size={18} strokeWidth={2.2} />
              ) : (
                <ShieldCheck size={18} strokeWidth={2.2} />
              )}
            </div>
            <div>
              <div className="modal-kicker">ISRO BAS // HITL SAFETY GATE</div>
              <h2 id="consent-modal-title" className="modal-title">
                {isReset ? 'Human Consent: Reset Sequence' : 'Human Consent: Start Sequence'}
              </h2>
            </div>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            disabled={submitting}
            aria-label="Close modal"
          >
            <X size={16} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {/* Action Impact Card */}
          <div className={`modal-action-card ${isReset ? 'action-warn' : 'action-info'}`}>
            <div className="modal-action-card-header">
              {isReset ? (
                <AlertTriangle size={15} className="action-icon-warn" />
              ) : (
                <Fingerprint size={15} className="action-icon-info" />
              )}
              <span className="modal-action-tag">
                {isReset ? 'CRITICAL PROCEDURE OVERRIDE' : 'MISSION EXECUTION PROTOCOL'}
              </span>
            </div>
            <p className="modal-action-desc">
              {isReset
                ? 'You are requesting a system-wide reset for the active sequence. All 8 autonomous vision agents, temporal buffers, and digital twin state will be reset to Step 0 (Standby). Video telemetry will rewind to the beginning.'
                : isComplete
                ? 'The previous procedure completed. You are authorizing a fresh sequence execution cycle. Autonomous action recognition and 3D skeleton tracking will engage from Step 0.'
                : 'You are authorizing real-time autonomous sequence validation. The vision pipeline will process multi-modal telemetry and track procedure step milestones.'}
            </p>
          </div>

          {/* ── Experiment Feed Dropdown (Dummy vs Live) ───────── */}
          <div className="modal-feed-select-card">
            <label htmlFor="modal-feed-select" className="modal-feed-label">
              <SlidersHorizontal size={13} />
              <span>EXPERIMENT SOURCE / FEED MODE:</span>
            </label>
            <div className="modal-select-wrapper">
              <select
                id="modal-feed-select"
                className="modal-select font-mono"
                value={feedMode}
                onChange={(e) => setFeedMode(e.target.value as FeedMode)}
                disabled={submitting}
              >
                <option value="dummy">Dummy — Pre-recorded Sequence (c1.mp4)</option>
                <option value="live">Live — Hardware Camera (Live Feed)</option>
              </select>
            </div>
            <div className={`modal-feed-hint ${feedMode === 'live' ? 'hint-live' : 'hint-dummy'}`}>
              {feedMode === 'dummy' ? (
                <>
                  <Film size={14} className="hint-icon" />
                  <span>
                    Processes recorded flight demonstration <strong>c1.mp4</strong> through the
                    autonomous sequence validation pipeline.
                  </span>
                </>
              ) : (
                <>
                  <Camera size={14} className="hint-icon" />
                  <span>
                    Engages <strong>Live Camera</strong> (webcam #0) for real-time astronaut
                    gesture and object interaction tracking.
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Sequence Details Table */}
          <div className="modal-meta-grid">
            <div className="modal-meta-row">
              <span className="modal-meta-key">Active Protocol</span>
              <span className="modal-meta-val font-mono">{experimentId || 'BAS-EXP-26174'}</span>
            </div>
            <div className="modal-meta-row">
              <span className="modal-meta-key">Selected Source</span>
              <span className="modal-meta-val font-mono">
                {feedMode === 'dummy' ? 'c1.mp4 (Dummy Feed)' : 'Camera #0 (Live Feed)'}
              </span>
            </div>
            <div className="modal-meta-row">
              <span className="modal-meta-key">Current Sequence</span>
              <span className="modal-meta-val">
                Step {currentStep}: <span className="font-mono">{currentStepName || 'STANDBY'}</span>
              </span>
            </div>
            <div className="modal-meta-row">
              <span className="modal-meta-key">Process State</span>
              <span className="modal-meta-val">
                {isComplete ? (
                  <span className="modal-state-badge badge-complete">COMPLETED (STOPPED)</span>
                ) : (
                  <span className="modal-state-badge badge-active">ACTIVE / STANDBY</span>
                )}
              </span>
            </div>
            <div className="modal-meta-row">
              <span className="modal-meta-key">Operator Authorization</span>
              <span className="modal-meta-val">Flight Director (Human-in-the-Loop)</span>
            </div>
          </div>

          {/* Interactive Human Consent Checkbox */}
          <label className={`modal-consent-checkbox-wrap ${confirmedConsent ? 'checked' : ''}`}>
            <input
              type="checkbox"
              className="modal-checkbox-input"
              checked={confirmedConsent}
              onChange={(e) => setConfirmedConsent(e.target.checked)}
              disabled={submitting}
            />
            <div className="modal-checkbox-custom">
              {confirmedConsent && (
                <svg viewBox="0 0 16 16" fill="none" className="modal-check-svg">
                  <polyline
                    points="3 8 7 12 13 4"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </div>
            <span className="modal-consent-text">
              <strong>I grant Human Consent</strong> and authorize execution using{' '}
              <strong>{feedMode === 'dummy' ? 'c1.mp4' : 'Live Camera'}</strong> under Bharatiya
              Antariksh Station sequence safety protocol #26174.
            </span>
          </label>
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button
            type="button"
            className="btn-modal btn-modal-cancel"
            onClick={onClose}
            disabled={submitting}
          >
            Abort / Cancel
          </button>

          <button
            type="button"
            className={`btn-modal ${
              isReset ? 'btn-modal-danger' : 'btn-modal-primary'
            } ${!confirmedConsent ? 'disabled' : ''}`}
            onClick={handleAuthorize}
            disabled={!confirmedConsent || submitting}
          >
            {submitting ? (
              <span className="modal-btn-inner">
                <span className="modal-spinner" />
                Authorizing...
              </span>
            ) : isReset ? (
              <span className="modal-btn-inner">
                <RotateCcw size={14} />
                Authorize & Reset ({feedMode === 'dummy' ? 'Dummy' : 'Live'})
              </span>
            ) : (
              <span className="modal-btn-inner">
                <Play size={14} fill="currentColor" />
                Authorize & Start ({feedMode === 'dummy' ? 'Dummy' : 'Live'})
              </span>
            )}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

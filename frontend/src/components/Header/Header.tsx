import { useState } from 'react';
import { FlaskConical } from 'lucide-react';
import type { TelemetryData } from '../../types/api';
import { EXPERIMENTS } from '../../mock/fallbackData';
import { API } from '../../services/api';

interface HeaderProps {
  telemetry?: TelemetryData;
  connected?: boolean;
}

export function Header({ telemetry, connected }: HeaderProps) {
  const [pendingSet, setPendingSet] = useState<string | null>(null);
  const activeSet = EXPERIMENTS.find(e => telemetry?.experiment_id && e.ids.includes(telemetry.experiment_id))?.set;

  const switchExperiment = async (set: string) => {
    if (set === activeSet || pendingSet) return;
    setPendingSet(set);
    try {
      await fetch(`${API.EXPERIMENT(set)}&t=${Date.now()}`);
      window.dispatchEvent(new CustomEvent('bas-source-switch', { detail: { experiment: set } }));
    } catch (err) {
      console.warn('Experiment switch failed:', err);
    } finally {
      // Backend applies the switch on its next frame; telemetry then confirms it
      setTimeout(() => setPendingSet(null), 1200);
    }
  };

  return (
    <header className="header">
      <div className="header-logo-wrapper">
        <img
          src="/sitara.png"
          alt="Sitara"
          className="header-logo"
        />
      </div>

      <nav className="exp-switch" aria-label="Experiment">
        <FlaskConical size={13} strokeWidth={2} className="exp-switch-icon" />
        {EXPERIMENTS.map(exp => (
          <button
            key={exp.set}
            type="button"
            className={[
              'exp-switch-btn',
              exp.set === activeSet ? 'exp-active' : '',
              exp.set === pendingSet ? 'exp-pending' : '',
            ].filter(Boolean).join(' ')}
            disabled={!connected || pendingSet !== null}
            onClick={() => switchExperiment(exp.set)}
            aria-pressed={exp.set === activeSet}
          >
            {exp.label}
          </button>
        ))}
      </nav>
    </header>
  );
}

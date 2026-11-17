import { ScanSearch, Hand } from 'lucide-react';
import type { TelemetryData, SceneObject } from '../../types/api';

interface SceneObjectsProps {
  telemetry: TelemetryData;
}

const INTERACTION_CLASS: Record<SceneObject['interaction'], string> = {
  HOLDING: 'so-holding',
  TOUCHING: 'so-touching',
  NEAR: 'so-near',
  NONE: '',
};

export function SceneObjects({ telemetry }: SceneObjectsProps) {
  const objects = telemetry.scene_objects ?? [];
  const interacting = objects.filter(o => o.interaction === 'HOLDING' || o.interaction === 'TOUCHING');

  return (
    <div className="panel scene-objects-panel">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <ScanSearch size={14} strokeWidth={2} style={{ color: 'var(--cyan)' }} />
          <span className="panel-title">Scene Objects &amp; Human Interaction</span>
        </div>
        <span className="panel-subtitle">{objects.length} objects in view</span>
      </div>

      <div className="so-body">
        <div className="so-interaction">
          <Hand size={14} strokeWidth={2} />
          {interacting.length > 0 ? (
            <span>
              Human is{' '}
              {interacting.map((o, i) => (
                <span key={`${o.label}-${i}`}>
                  {i > 0 && ', '}
                  <strong>{o.interaction.toLowerCase()}</strong> {o.label}
                </span>
              ))}
            </span>
          ) : (
            <span className="so-muted">No object in hand</span>
          )}
        </div>

        <div className="so-chips">
          {objects.length === 0 && <span className="so-muted">Waiting for detections…</span>}
          {objects.map((o, i) => (
            <span key={`${o.label}-${i}`} className={`so-chip ${INTERACTION_CLASS[o.interaction]}`}>
              <span className="so-chip-label">{o.label}</span>
              <span className="so-chip-conf font-mono">{Math.round(o.conf * 100)}%</span>
              {o.interaction !== 'NONE' && <span className="so-chip-tag">{o.interaction}</span>}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

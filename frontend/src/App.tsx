import { useTelemetry } from './hooks/useTelemetry';
import { Header } from './components/Header/Header';
import { Camera1 } from './components/Camera1/Camera1';
import { Camera2 } from './components/Camera2/Camera2';
import { ProcessTimeline } from './components/ProcessTimeline/ProcessTimeline';
import { AnomalyDetection } from './components/AnomalyDetection/AnomalyDetection';
import { Logs } from './components/Logs/Logs';
import { SceneObjects } from './components/SceneObjects/SceneObjects';

export default function App() {
  const { data: telemetry, connected } = useTelemetry();

  return (
    <div className="app-root">
      <Header telemetry={telemetry} connected={connected} />

      <main className="main-grid">
        {/* Row 1: Camera panels side by side */}
        <div className="cameras-row">
          <Camera1 telemetry={telemetry} />
          <Camera2 />
        </div>

        {/* Row 2: Process timeline full width */}
        <ProcessTimeline telemetry={telemetry} />

        {/* Row 3: Open-vocabulary scene objects & human interaction */}
        <SceneObjects telemetry={telemetry} />

        {/* Row 4: Anomaly + Logs side by side */}
        <div className="bottom-row">
          <AnomalyDetection telemetry={telemetry} />
          <Logs telemetry={telemetry} />
        </div>
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// BAS-HAR System — API Type Definitions
// These mirror the exact JSON shapes returned by the backend.
// Do NOT add fields that the backend does not provide.
// ─────────────────────────────────────────────────────────────────────────────

// ── /telemetry ───────────────────────────────────────────────────────────────
export interface TelemetryData {
  frame_id: number;
  fps: number;
  latency_ms: number;
  step: number;
  step_name: string;
  is_step_correct: boolean;
  step_verdict: string;
  what_i_am_doing: string;
  what_i_have_to_do: string;
  instruction: string;
  anomaly: string; // "NONE" | anomaly code string
  transition_event: string | null;
  source_type: string; // "LIVE_WEBCAM" | "RED_YELLOW" | clip name
  experiment_id: string; // e.g. "BAS-EXP-26174"
  debounce?: number;
  debounce_max?: number;
  lid_angle?: number;
}

// ── /api/digital_twin ────────────────────────────────────────────────────────
export interface Vec3 {
  x?: number;
  y?: number;
  z?: number;
}

export interface EntityData {
  state: string;
  pos_rack: [number, number, number];
  is_inside_container: boolean;
}

export interface JointData {
  pos_rack: [number, number, number];
  confidence: number;
}

export interface DigitalTwinData {
  version: string;
  step: string;
  activity: string;
  rack: {
    label: string;
    bounds: [number, number, number];
  };
  container: {
    lid_angle_deg: number;
    is_open: boolean;
    origin: [number, number, number];
  };
  entities: Record<string, EntityData>;
  astronaut: {
    joints: Record<string, JointData>;
    keypoints_2d?: number[][];
    elbow_angle_deg: number;
    rom_violated: boolean;
  };
}

// ── /api/session_actions ─────────────────────────────────────────────────────
export interface ActionEntry {
  step: number;
  step_name: string;
  what_i_am_doing: string;
  what_i_have_to_do: string;
  start_time_sec: number;
  end_time_sec: number;
  duration_sec: number;
  start_frame: number;
  end_frame: number;
  lid_angle_deg: number;
  compliance: string;
}

export interface SessionSummary {
  total_steps_executed: number;
  anomalies_flagged: number;
  anomaly_details: string[];
  compliance_status: string;
}

export interface SessionActionsData {
  session_id: string;
  experiment_id: string;
  procedure_name: string;
  start_time: string;
  last_updated: string;
  elapsed_seconds: number;
  total_actions_recorded: number;
  summary: SessionSummary;
  timeline: ActionEntry[];
}

// ── Frontend-only log entry ───────────────────────────────────────────────────
export type LogLevel = 'INFO' | 'SUCCESS' | 'WARNING' | 'ERROR';

export interface LogEntry {
  id: string;
  timestamp: string; // HH:MM:SS
  level: LogLevel;
  message: string;
}

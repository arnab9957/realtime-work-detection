// ─────────────────────────────────────────────────────────────────────────────
// BAS-HAR System — Fallback / Mock data
// Used when backend is unreachable or returns empty data.
// CPU/GPU/RAM/VRAM are frontend-simulated since backend doesn't expose them.
// ─────────────────────────────────────────────────────────────────────────────

import type { TelemetryData, DigitalTwinData, SessionActionsData } from '../types/api';

export const FALLBACK_TELEMETRY: TelemetryData = {
  frame_id: 0,
  fps: 0,
  latency_ms: 0,
  step: 0,
  step_name: 'IDLE',
  is_step_correct: true,
  step_verdict: 'AWAITING STREAM',
  what_i_am_doing: 'IDLE',
  what_i_have_to_do: 'Waiting for system...',
  instruction: 'Waiting for system...',
  anomaly: 'NONE',
  transition_event: null,
  source_type: 'UNKNOWN',
  experiment_id: 'BAS-EXP-26174',
};

export const FALLBACK_DIGITAL_TWIN: DigitalTwinData = {
  version: '1.0',
  step: 'IDLE',
  activity: 'IDLE',
  rack: {
    label: 'BAS-03 Payload Experiment Rack',
    bounds: [1.0, 0.8, 1.2],
  },
  container: {
    lid_angle_deg: 0,
    is_open: false,
    origin: [0, 0, 0],
  },
  entities: {
    container_box: {
      state: 'CLOSED',
      pos_rack: [0, 0, 0],
      is_inside_container: false,
    },
    component_box: {
      state: 'DOCKED',
      pos_rack: [0.05, 0.05, 0],
      is_inside_container: true,
    },
  },
  astronaut: {
    joints: {
      shoulder: { pos_rack: [0.15, 0.5, 0.3], confidence: 0.9 },
      elbow: { pos_rack: [0.2, 0.35, 0.3], confidence: 0.9 },
      wrist: { pos_rack: [0.2, 0.2, 0.3], confidence: 0.9 },
    },
    elbow_angle_deg: 120,
    rom_violated: false,
  },
};

export const FALLBACK_SESSION: SessionActionsData = {
  session_id: 'SES-OFFLINE',
  experiment_id: 'BAS-EXP-26174',
  procedure_name: 'Box Object Extraction & Return Procedure',
  start_time: new Date().toISOString(),
  last_updated: new Date().toISOString(),
  elapsed_seconds: 0,
  total_actions_recorded: 0,
  summary: {
    total_steps_executed: 0,
    anomalies_flagged: 0,
    anomaly_details: [],
    compliance_status: 'AWAITING SESSION',
  },
  timeline: [],
};

// ── Frontend step label mapping ───────────────────────────────────────────────
// Maps backend FSM step names → human-readable UI labels
// Backend step names are NOT modified; only the UI display label is provided here.
export const STEP_LABELS: Record<string, string> = {
  IDLE: 'Man standing in front of box container',
  APPROACH: 'Man approaching the box container',
  HOLD_CONTAINER: 'Man holding the box container',
  OPEN_BOX: 'Man opening the box container',
  CONTAINER_OPEN: 'Man opening the box container',
  BOX_OPENED: 'Man opening the box container',
  RED_EXTRACTED: 'Man taking out red box from container',
  OBJECT_EXTRACTED: 'Man taking out object from container',
  TAKE_OUT_OBJECT: 'Man taking out red box from container',
  YELLOW_EXTRACTED: 'Man taking out yellow box from container',
  TAKE_OUT_YELLOW: 'Man taking out yellow box from container',
  OBJECTS_RETURNED: 'Man returning yellow and red boxes into container',
  RETURN_OBJECT: 'Man returning box into container',
  RETURN_YELLOW: 'Man returning yellow box into container',
  BOX_CLOSED: 'Man closing the box container',
  COMPLETE: 'Man standing — procedure complete',
  DUAL_COMPLETE: 'Dual experiment procedure complete',
};

export interface ProcessStep {
  key: string;
  aliases?: string[];
  label: string;
  shortLabel: string;
}

// Dual-box procedure (ISRO Red-Yellow / BAS-EXP-26174 / BAS-EXP-RED-YELLOW)
export const RED_YELLOW_STEPS: ProcessStep[] = [
  { key: 'IDLE', aliases: ['STANDBY'], label: 'Standby / Box Closed', shortLabel: 'Standby' },
  { key: 'CONTAINER_OPEN', aliases: ['BOX_OPENED', 'OPEN_BOX'], label: 'Opening Container Box', shortLabel: 'Open Box' },
  { key: 'RED_EXTRACTED', aliases: ['OBJECT_EXTRACTED', 'TAKE_OUT_OBJECT'], label: 'Extract Red Box First', shortLabel: 'Extract Red' },
  { key: 'YELLOW_EXTRACTED', aliases: ['TAKE_OUT_YELLOW'], label: 'Extract Yellow Box Second', shortLabel: 'Extract Yellow' },
  { key: 'OBJECTS_RETURNED', aliases: ['RETURN_OBJECT', 'RETURN_YELLOW'], label: 'Return Yellow & Red Boxes', shortLabel: 'Return Boxes' },
  { key: 'COMPLETE', aliases: ['BOX_CLOSED', 'DUAL_COMPLETE'], label: 'Close Box & Complete', shortLabel: 'Complete' },
];

// Single-box procedure (BAS-EXP-BOX-RETURN)
export const BOX_RETURN_STEPS: ProcessStep[] = [
  { key: 'IDLE', aliases: ['STANDBY'], label: 'Initialize Workspace', shortLabel: 'Standby' },
  { key: 'BOX_OPENED', aliases: ['CONTAINER_OPEN', 'OPEN_BOX'], label: 'Opening Box', shortLabel: 'Open Box' },
  { key: 'OBJECT_EXTRACTED', aliases: ['RED_EXTRACTED', 'TAKE_OUT_OBJECT'], label: 'Take Out Object', shortLabel: 'Extract Object' },
  { key: 'OBJECT_RETURNED', aliases: ['RETURN_OBJECT'], label: 'Return Object into Box', shortLabel: 'Return Object' },
  { key: 'COMPLETE', aliases: ['BOX_CLOSED'], label: 'Close Box & Complete', shortLabel: 'Complete' },
];

// Default ordered process steps for the timeline display
export const PROCESS_STEPS = RED_YELLOW_STEPS;

// Simulated system resource metrics (frontend-only; backend doesn't expose these)
export function getSimulatedSystemMetrics() {
  return {
    cpu: Math.floor(28 + Math.random() * 20),
    gpu: Math.floor(55 + Math.random() * 20),
    ram: Math.floor(42 + Math.random() * 15),
    vram: Math.floor(50 + Math.random() * 20),
  };
}

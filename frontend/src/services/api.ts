const BASE_URL = 'http://localhost:8080';
// Browser-camera ingest served by main.py --source browser (dashboard port + 1)
const CAMERA_INGEST_URL = 'http://localhost:8081';

export const API = {
  TELEMETRY:      `${BASE_URL}/telemetry`,
  DIGITAL_TWIN:   `${BASE_URL}/api/digital_twin`,
  SESSION_ACTIONS:`${BASE_URL}/api/session_actions`,
  STREAM:         `${BASE_URL}/stream`,
  SNAPSHOT:       `${BASE_URL}/snapshot`,
  TWIN_STREAM:    `${BASE_URL}/twin_stream`,
  TWIN_SNAPSHOT:  `${BASE_URL}/twin_snapshot`,
  RESET:          `${BASE_URL}/reset`,
  START:          `${BASE_URL}/start`,
  SOURCE:         (set: string) => `${BASE_URL}/api/source?set=${encodeURIComponent(set)}`,
  EXPERIMENT:     (set: string) => `${BASE_URL}/api/experiment?set=${encodeURIComponent(set)}`,
  CAMERA_FRAME:   `${CAMERA_INGEST_URL}/frame`,
};

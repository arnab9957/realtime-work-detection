const BASE_URL = 'http://localhost:8080';

export const API = {
  TELEMETRY:      `${BASE_URL}/telemetry`,
  DIGITAL_TWIN:   `${BASE_URL}/api/digital_twin`,
  SESSION_ACTIONS:`${BASE_URL}/api/session_actions`,
  STREAM:         `${BASE_URL}/stream`,
  SNAPSHOT:       `${BASE_URL}/snapshot`,
  RESET:          `${BASE_URL}/reset`,
  SOURCE:         (set: string) => `${BASE_URL}/api/source?set=${encodeURIComponent(set)}`,
  EXPERIMENT:     (set: string) => `${BASE_URL}/api/experiment?set=${encodeURIComponent(set)}`,
};

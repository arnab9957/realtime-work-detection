"""
BAS Autonomous HAR System - Agent 7: Reasoning & Guidance Agent
Interprets procedural state, produces next-step astronaut instructions,
and formulates immediate priority voice alerts and recovery prompts upon wrong moves or anomalies.
"""

import json
import time
from typing import Optional, Tuple
from src.core.types import FSMStep, AnomalyType


class ReasoningAgent:
    """Provides conversational mission guidance, next-step recommendations, and recovery advice."""

    def __init__(self, config_path: str = "configs/experiment_fsm.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.step_instructions = {}
        for s_idx_str, s_info in self.config.get("states", {}).items():
            try:
                step_enum = FSMStep(int(s_idx_str))
                self.step_instructions[step_enum] = s_info["instruction"]
            except ValueError:
                pass

        self.last_guided_step: Optional[FSMStep] = None
        self.last_anomaly_alert: Optional[AnomalyType] = None
        self.is_in_wrong_move: bool = False
        self.last_spoken_wrong_move: str = ""
        self.last_wrong_move_time: float = 0.0

    def reset(self):
        """Resets guidance memory when restarting or looping."""
        self.last_guided_step = None
        self.last_anomaly_alert = None
        self.is_in_wrong_move = False
        self.last_spoken_wrong_move = ""
        self.last_wrong_move_time = 0.0

    def evaluate_guidance(
        self,
        current_step: FSMStep,
        anomaly: AnomalyType,
        transition_event: Optional[str],
        vlm_explanation: Optional[str] = None,
        is_step_correct: bool = True,
        step_verdict: str = "",
        anomaly_message: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Determines current on-screen instruction and any pending spoken alert.
        Returns:
            - active_instruction: Text instruction displayed in the GUI HUD
            - voice_alert_to_speak: Voice utterance string if audio should be triggered, else None
        """
        voice_alert: Optional[str] = None
        instruction = self.step_instructions.get(current_step, "Awaiting instructions.")
        now = time.time()

        # =====================================================================
        # 1. Handle Critical Anomalies & Wrong Moves (Highest Priority)
        # =====================================================================
        is_wrong_move = (not is_step_correct) or (anomaly != AnomalyType.NONE)

        if is_wrong_move:
            self.last_anomaly_alert = anomaly

            # Formulate clear, urgent spoken alert
            if anomaly_message and anomaly_message.strip():
                if not anomaly_message.startswith("Warning: Wrong move"):
                    voice_alert = f"Warning: Wrong move! {anomaly_message.replace('Warning: ', '')}"
                else:
                    voice_alert = anomaly_message
            elif vlm_explanation and vlm_explanation != "None":
                voice_alert = f"Warning: Wrong move! {vlm_explanation}"
            else:
                anom_key = anomaly.value if hasattr(anomaly, "value") else str(anomaly)
                anom_dict = self.config.get("anomalies", {})
                anom_info = anom_dict.get(anom_key) or anom_dict.get(getattr(anomaly, "name", ""), {})
                voice_alert = anom_info.get("alert_tts")
            if not voice_alert:
                voice_alert = "Warning: Wrong move! Procedural deviation detected."

            # Formulate on-screen HUD instruction
            clean_display = voice_alert.replace("Warning: Wrong move! ", "").replace("Warning: ", "")
            instruction = f"WRONG MOVE: {clean_display}"

            # Speak immediately on first detection, or repeat every 4.0s if persistent
            if not self.is_in_wrong_move or voice_alert != self.last_spoken_wrong_move or (now - self.last_wrong_move_time) > 4.0:
                self.is_in_wrong_move = True
                self.last_spoken_wrong_move = voice_alert
                self.last_wrong_move_time = now
                return instruction, voice_alert
            else:
                return instruction, None

        # =====================================================================
        # 2. Recovery from Wrong Move -> Re-orient with Next Step
        # =====================================================================
        if self.is_in_wrong_move:
            self.is_in_wrong_move = False
            self.last_spoken_wrong_move = ""
            self.last_wrong_move_time = 0.0
            self.last_anomaly_alert = None
            # Force speaking the current step to immediately re-orient the operator
            self.last_guided_step = None

        # Reset anomaly memory once resolved
        if anomaly == AnomalyType.NONE:
            self.last_anomaly_alert = None

        # =====================================================================
        # 3. Handle State Transitions / Next-Step Spoken Guidance
        # =====================================================================
        if transition_event is not None or current_step != self.last_guided_step:
            self.last_guided_step = current_step
            voice_alert = instruction

        return instruction, voice_alert

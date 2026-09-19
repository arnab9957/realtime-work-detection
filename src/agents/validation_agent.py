"""
BAS Autonomous HAR System - Agent 6: Validation Agent (Safety-Critical FSM)
Authoritative, deterministic finite state machine validating procedural compliance
with adaptive temporal debouncing, Local LLM consensus verification, and robust anomaly protection.
"""

import json
import time
from typing import Dict, Tuple, Optional, Any, List
from src.core.types import (
    FSMStep,
    AnomalyType,
    ExperimentObject,
    EntityState,
    HOIInteraction,
    HOIAction,
    AstronautPose3D,
    FORBIDDEN_EVENT_CONDITIONS
)

try:
    import torch
except ImportError:
    torch = None


class ValidationAgent:
    """Deterministic Sequence Validator and Safety Gatekeeper with Local LLM Consensus."""

    def __init__(self, config_path: str = "configs/box_return_fsm.json"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.experiment_id: str = self.config.get("experiment_id", "BAS-EXP-BOX-RETURN")
        self.debounce_required: int = self.config.get("debounce_frames", 6)
        self.current_step: FSMStep = FSMStep.IDLE
        self.debounce_counter: int = 0
        self.candidate_step: Optional[FSMStep] = None

        self.anomaly_status: AnomalyType = AnomalyType.NONE
        self.anomaly_message: str = ""
        self.anomaly_debounce_counter: int = 0
        self.step_start_time: float = time.time()

        # Step Correctness Tracking (Nominal vs Anomaly)
        self.is_step_correct: bool = True
        self.step_verdict: str = "CORRECT (NOMINAL)"
        self.vlm_anomaly_explanation: Optional[str] = None

        # --- GAP 4: Config-driven forbidden_events enforcement ---
        self.forbidden_events_map: Dict[int, List[str]] = {}
        for step_idx_str, step_info in self.config.get("states", {}).items():
            forbidden = step_info.get("forbidden_events", [])
            if forbidden:
                self.forbidden_events_map[int(step_idx_str)] = forbidden

        # --- GAP 6: Per-step stall timeouts from config ---
        self.stall_timeout_sec: float = float(
            self.config.get("anomalies", {})
            .get("STALL_TIMEOUT", {})
            .get("timeout_seconds", 60.0)
        )
        self.per_step_timeouts: Dict[int, Optional[float]] = {}
        for step_idx_str, step_info in self.config.get("states", {}).items():
            t = step_info.get("timeout_seconds")
            self.per_step_timeouts[int(step_idx_str)] = float(t) if t is not None else None

        # --- GAP 5: Backward regression detection ---
        self.highest_committed_step: int = 0
        self.regression_debounce: int = 0
        self.regression_debounce_required: int = 10  # ~333ms @ 30fps

        # --- GAP 2: Ergonomic envelope checking ---
        self.ergonomic_envelopes: Dict[int, Dict[str, Any]] = {}
        for step_idx_str, step_info in self.config.get("states", {}).items():
            envelope = step_info.get("ergonomic_envelope")
            if envelope:
                self.ergonomic_envelopes[int(step_idx_str)] = envelope

        # --- GAP 3: HOI event buffer for retroactive VLM validation ---
        self.event_buffer: List[Dict[str, Any]] = []
        self.event_buffer_max: int = 60  # ~2 seconds of events @ 30fps

        # --- GAP 1: Temporal action sequence anomaly classifier ---
        self.har_classifier = None
        self.har_anomaly_threshold: float = 0.65
        if torch is not None:
            har_model_path = "models/har_sequence_classifier.pt"
            try:
                from tools.train_experiment import ProceduralHARClassifier
                self.har_classifier = ProceduralHARClassifier(input_dim=8, num_classes=5)
                self.har_classifier.load_state_dict(
                    torch.load(har_model_path, map_location="cpu", weights_only=True)
                )
                self.har_classifier.eval()
            except Exception:
                self.har_classifier = None  # Graceful degradation

    def load_protocol(self, config_path: str):
        """Dynamically switches the active procedural FSM protocol."""
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.config_path = config_path
        self.experiment_id = self.config.get("experiment_id", "BAS-EXP-BOX-RETURN")
        self.debounce_required = self.config.get("debounce_frames", 6)
        # Rebuild config-driven lookup tables
        self.forbidden_events_map = {}
        for step_idx_str, step_info in self.config.get("states", {}).items():
            forbidden = step_info.get("forbidden_events", [])
            if forbidden:
                self.forbidden_events_map[int(step_idx_str)] = forbidden
        self.stall_timeout_sec = float(
            self.config.get("anomalies", {})
            .get("STALL_TIMEOUT", {})
            .get("timeout_seconds", 60.0)
        )
        self.per_step_timeouts = {}
        for step_idx_str, step_info in self.config.get("states", {}).items():
            t = step_info.get("timeout_seconds")
            self.per_step_timeouts[int(step_idx_str)] = float(t) if t is not None else None
        self.ergonomic_envelopes = {}
        for step_idx_str, step_info in self.config.get("states", {}).items():
            envelope = step_info.get("ergonomic_envelope")
            if envelope:
                self.ergonomic_envelopes[int(step_idx_str)] = envelope
        self.reset()

    def reset(self):
        """Resets the state machine back to IDLE (Step 0) for a fresh real-time test run."""
        self.current_step = FSMStep.IDLE
        self.debounce_counter = 0
        self.candidate_step = None
        self.anomaly_status = AnomalyType.NONE
        self.anomaly_message = ""
        self.anomaly_debounce_counter = 0
        self.step_start_time = time.time()
        self.is_step_correct = True
        self.step_verdict = "CORRECT (NOMINAL)"
        self.vlm_anomaly_explanation = None
        # Gap 5: Regression tracking
        self.highest_committed_step = 0
        self.regression_debounce = 0
        # Gap 3: Event buffer
        self.event_buffer.clear()

    def evaluate_step(
        self,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        active_hoi: list,
        current_frame: int,
        llm_verification: Optional[Dict[str, Any]] = None,
        action_history: Optional[List[Dict]] = None,
        astronaut_pose: Optional[AstronautPose3D] = None
    ) -> Tuple[FSMStep, int, AnomalyType, str, Optional[str]]:
        """
        Evaluates current physical state against procedural state machine with Local LLM consensus.
        Returns:
            - current_step: committed FSMStep
            - debounce_count: frames accumulated towards candidate transition
            - anomaly: AnomalyType
            - anomaly_msg: diagnostic string
            - transition_event: string name if a new state was committed on this frame
        """
        now = time.time()
        transition_committed: Optional[str] = None

        # --- GAP 3: Record frame into event buffer for retroactive validation ---
        self._record_event(objects, lid_angle, active_hoi, current_frame)

        # --- GAP 6: Per-step stall timeout (config-driven) ---
        if self.current_step not in (FSMStep.IDLE, FSMStep.COMPLETE):
            step_timeout = self.per_step_timeouts.get(int(self.current_step), self.stall_timeout_sec)
            if step_timeout is not None and (now - self.step_start_time) > step_timeout:
                self.anomaly_status = AnomalyType.STALL_TIMEOUT
                self.anomaly_message = (
                    f"Activity paused for {int(now - self.step_start_time)}s at "
                    f"{self.current_step.name}. Expected completion within {int(step_timeout)}s."
                )
                self.is_step_correct = False
                self.step_verdict = "PROCEDURAL ERROR [STALL_TIMEOUT]"
                return self.current_step, self.debounce_counter, self.anomaly_status, self.anomaly_message, None

        # Extract LLM/VLM verified step & anomaly diagnostic if available
        llm_step_val = None
        llm_confidence = 0.0
        llm_what_wrong = None
        llm_verdict = "NOMINAL"
        if llm_verification:
            llm_step_val = llm_verification.get("verified_step")
            llm_confidence = float(llm_verification.get("confidence", 0.0))
            llm_verdict = llm_verification.get("anomaly_verdict", "NOMINAL")
            raw_wrong = llm_verification.get("what_is_wrong")
            if raw_wrong and raw_wrong != "None":
                llm_what_wrong = raw_wrong

        # =================================================================
        # SIDE-BY-SIDE VLM STEP GUARDIAN: Detect Unlisted or Future Steps
        # =================================================================
        max_protocol_step = 5 if self.experiment_id in ("BAS-EXP-26174", "BAS-EXP-RED-YELLOW") else 4
        if llm_step_val is not None and llm_confidence >= 0.70:
            if llm_step_val > max_protocol_step or llm_step_val < 0:
                # Step not included in main steps
                self.anomaly_status = AnomalyType.ERROR_SEQ
                self.anomaly_message = f"Warning: Wrong move! Unrecognized procedural step {llm_step_val} detected."
                self.is_step_correct = False
                self.step_verdict = f"WRONG STEP: Step {llm_step_val} not in protocol [ERROR_SEQ]"
                return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
            elif llm_step_val > int(self.current_step) + 1 and current_frame >= 45:
                # Future step performed prematurely!
                self.anomaly_status = AnomalyType.ERROR_SKIP
                detail = llm_what_wrong if (llm_what_wrong and "future" in llm_what_wrong.lower()) else f"Future step {llm_step_val} detected prematurely"
                self.anomaly_message = f"Warning: Wrong move! Future step detected prematurely. Please complete {self.current_step.name} first."
                self.is_step_correct = False
                self.step_verdict = f"WRONG STEP: {detail} [ERROR_SKIP]"
                return self.current_step, 0, self.anomaly_status, self.anomaly_message, None

        if llm_verdict == "PROCEDURAL_ERROR" and llm_what_wrong and llm_what_wrong != "None" and llm_confidence >= 0.75:
            self.vlm_anomaly_explanation = llm_what_wrong
            if not self.anomaly_message:
                self.anomaly_message = f"Warning: Wrong move! {llm_what_wrong}"

        # --- GAP 4: Generic config-driven forbidden_events enforcement ---
        forbidden_result = self._check_forbidden_events(objects, lid_angle, active_hoi)
        if forbidden_result:
            anom_type, anom_msg = forbidden_result
            # Only escalate if not already caught by hardcoded or VLM checks
            if self.anomaly_status == AnomalyType.NONE:
                self.anomaly_status = anom_type
                self.anomaly_message = anom_msg
                self.is_step_correct = False
                self.step_verdict = f"PROCEDURAL ERROR [{self.anomaly_status.value}]"
                return self.current_step, self.debounce_counter, self.anomaly_status, self.anomaly_message, None
                self.is_step_correct = False
                self.step_verdict = f"PROCEDURAL ERROR [{anom_type.value}]"

        # --- GAP 3: Retroactive VLM validation from event buffer ---
        if llm_verification and llm_verification.get("anomaly_verdict") == "PROCEDURAL_ERROR":
            retro_result = self._retroactive_validate(llm_verification)
            if retro_result and self.anomaly_status == AnomalyType.NONE:
                self.anomaly_status, self.anomaly_message = retro_result
                self.is_step_correct = False
                self.step_verdict = f"RETROACTIVE VIOLATION [{self.anomaly_status.value}]"
                return self.current_step, self.debounce_counter, self.anomaly_status, self.anomaly_message, None

        # =================================================================
        # BRANCH A: DUAL-BOX RED & YELLOW EXPERIMENT (BAS-EXP-RED-YELLOW / BAS-EXP-26174)
        # Sequence:
        # Step 0: IDLE
        # Step 1: CONTAINER_OPEN
        # Step 2: RED_EXTRACTED (Red must be extracted first)
        # Step 3: YELLOW_EXTRACTED (Yellow extracted second)
        # Step 4: OBJECTS_RETURNED (Yellow and red boxes returned into container)
        # Step 5: COMPLETE / BOX_CLOSED (Container box closed)
        # =================================================================
        if self.experiment_id in ("BAS-EXP-26174", "BAS-EXP-RED-YELLOW"):
            # Clear VLM-induced ERROR_SKIP in Branch A if it's no longer actively triggered by the VLM Guardian
            if self.anomaly_status == AnomalyType.ERROR_SKIP:
                self.anomaly_status = AnomalyType.NONE
                self.anomaly_message = ""
                self.is_step_correct = True
                self.step_verdict = "CORRECT (NOMINAL)"
                self.vlm_anomaly_explanation = None

            # Step 0: IDLE -> Awaiting Container Opening
            if self.current_step == FSMStep.IDLE:
                is_opening = (
                    lid_angle >= 40.0
                    or (llm_step_val is not None and llm_step_val >= 1 and llm_confidence >= 0.75 and lid_angle >= 14.0)
                )
                if is_opening:
                    self._accumulate_debounce(FSMStep.CONTAINER_OPEN)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.CONTAINER_OPEN
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "CONTAINER_OPENED"
                else:
                    self._reset_debounce()

            # Step 1: CONTAINER_OPEN -> Extract Red Box
            elif self.current_step == FSMStep.CONTAINER_OPEN:
                yellow_obj = objects.get("yellow_box")
                red_obj = objects.get("red_box")

                # Sequence protection: Yellow cannot be extracted before Red
                yellow_violation = False
                if yellow_obj and (not yellow_obj.is_inside_container or yellow_obj.state == EntityState.EXTRACTED):
                    if not red_obj or red_obj.is_inside_container:
                        yellow_violation = True

                if yellow_violation:
                    self.anomaly_status = AnomalyType.ERROR_SEQ
                    self.anomaly_message = "Warning: Wrong move! Red box must be extracted before yellow box. Please return the yellow box."
                    self.is_step_correct = False
                    self.step_verdict = "PROCEDURAL ERROR [ERROR_SEQ]"
                    return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    if self.anomaly_status == AnomalyType.ERROR_SEQ:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "CORRECT (NOMINAL)"

                # Check Red Box Extracted
                red_extracted = False
                if red_obj and (not red_obj.is_inside_container or red_obj.state == EntityState.EXTRACTED):
                    red_extracted = True

                if red_extracted:
                    self._accumulate_debounce(FSMStep.RED_EXTRACTED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.RED_EXTRACTED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "RED_BOX_EXTRACTED"
                else:
                    self._reset_debounce()

            # Step 2: RED_EXTRACTED -> Extract Yellow Box
            elif self.current_step == FSMStep.RED_EXTRACTED:
                # Check Yellow Box Extracted
                yellow_extracted = False
                yellow_obj = objects.get("yellow_box")
                if yellow_obj and (not yellow_obj.is_inside_container or yellow_obj.state == EntityState.EXTRACTED):
                    yellow_extracted = True

                if yellow_extracted:
                    self._accumulate_debounce(FSMStep.YELLOW_EXTRACTED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.YELLOW_EXTRACTED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "YELLOW_BOX_EXTRACTED"
                else:
                    self._reset_debounce()

            # Step 3: YELLOW_EXTRACTED -> Return Yellow and Red Boxes into Container
            elif self.current_step == FSMStep.YELLOW_EXTRACTED:
                # Check both boxes returned into container
                objects_returned = False
                yellow_obj = objects.get("yellow_box")
                red_obj = objects.get("red_box")
                if (yellow_obj and yellow_obj.is_inside_container) and (red_obj and red_obj.is_inside_container):
                    is_grasped = any(h.object_name in ("red_box", "yellow_box") and h.action in (HOIAction.GRASP, HOIAction.CONTACT) for h in active_hoi)
                    if not is_grasped:
                        objects_returned = True

                if objects_returned:
                    self._accumulate_debounce(FSMStep.OBJECTS_RETURNED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.OBJECTS_RETURNED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "OBJECTS_RETURNED"
                else:
                    self._reset_debounce()

            # Step 4: OBJECTS_RETURNED -> Close Container Box
            elif self.current_step == FSMStep.OBJECTS_RETURNED:
                box_closed = (
                    lid_angle <= 18.0
                    and ("container_lid" not in objects or objects["container_lid"].bbox is None)
                )
                if box_closed:
                    self._accumulate_debounce(FSMStep.BOX_CLOSED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.BOX_CLOSED
                        self.candidate_step = None
                        self.debounce_counter = self.debounce_required
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "BOX_CLOSED"
                else:
                    self._reset_debounce()

            # Step 5: COMPLETE / BOX_CLOSED -> Terminal state / Looping
            elif self.current_step in (FSMStep.BOX_CLOSED, FSMStep.DUAL_COMPLETE):
                self.debounce_counter = self.debounce_required
                self.is_step_correct = True
                self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
                if (now - self.step_start_time) >= 3.0 and lid_angle >= 28.0:
                    self.current_step = FSMStep.CONTAINER_OPEN
                    self.candidate_step = None
                    self.step_start_time = now
                    self.anomaly_status = AnomalyType.NONE
                    self.anomaly_message = ""
                    transition_committed = "CONTAINER_REOPENED"

        # =================================================================
        # BRANCH B: BOX OBJECT EXTRACTION & RETURN PROCEDURE (BAS-EXP-BOX-RETURN)
        # Sequence: IDLE -> OPEN BOX -> EXTRACT OBJECT -> RETURN OBJECT -> CLOSE BOX
        # =================================================================
        else:
            # Step 0: IDLE -> Awaiting Box Open
            if self.current_step == FSMStep.IDLE:
                is_opening = (lid_angle >= 45.0)
                if is_opening:
                    self._accumulate_debounce(FSMStep.BOX_OPENED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.BOX_OPENED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"
                        transition_committed = "BOX_OPENED"
                else:
                    self._reset_debounce()

            # Step 1: BOX_OPENED -> Awaiting Object Extraction
            elif self.current_step == FSMStep.BOX_OPENED:
                # Check premature close anomaly
                if lid_angle <= 10.0:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 8:
                        if not (llm_verification and llm_verification.get("anomaly_verdict") == "NOMINAL"):
                            self.anomaly_status = AnomalyType.ERROR_SKIP
                            self.anomaly_message = "Warning: Wrong move! Step skipped. Please extract the object before closing the box."
                            self.is_step_correct = False
                            if llm_what_wrong:
                                self.vlm_anomaly_explanation = llm_what_wrong
                                self.step_verdict = f"WRONG STEP: {llm_what_wrong} [ERROR_SKIP]"
                            else:
                                self.step_verdict = "WRONG STEP: Box closed before extracting object! [ERROR_SKIP]"
                            return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SKIP:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"

                # Check if manipulable object is physically extracted outside the box
                extracted = False
                EXCLUDED_NON_PAYLOAD = {"container_box", "container_lid", "operator_hand", "human_body", "person"}
                for name, obj in objects.items():
                    if name in ("component_box", "red_box", "yellow_box") or (name not in EXCLUDED_NON_PAYLOAD):
                        if not obj.is_inside_container or obj.state == EntityState.EXTRACTED:
                            extracted = True
                            break

                if not extracted and any(h.action == HOIAction.EXTRACT for h in active_hoi):
                    extracted = True

                if extracted:
                    self._accumulate_debounce(FSMStep.OBJECT_EXTRACTED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.OBJECT_EXTRACTED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.anomaly_debounce_counter = 0
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"
                        transition_committed = "OBJECT_EXTRACTED"
                else:
                    self._reset_debounce()

            # Step 2: OBJECT_EXTRACTED -> Awaiting Object Return into Box
            elif self.current_step == FSMStep.OBJECT_EXTRACTED:
                # Check premature close anomaly
                if lid_angle <= 10.0:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 8:
                        if not (llm_verification and llm_verification.get("anomaly_verdict") == "NOMINAL"):
                            self.anomaly_status = AnomalyType.ERROR_SEQ
                            self.anomaly_message = "Warning: Wrong move! The object has not been returned to the box."
                            self.is_step_correct = False
                            if llm_what_wrong:
                                self.vlm_anomaly_explanation = llm_what_wrong
                                self.step_verdict = f"WRONG STEP: {llm_what_wrong} [ERROR_SEQ]"
                            else:
                                self.step_verdict = "WRONG STEP: Object not returned into box before closing! [ERROR_SEQ]"
                            return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SEQ:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"

                # Check if object has physically returned back inside container cavity
                all_inside = True
                found_target = False
                EXCLUDED_NON_PAYLOAD = {"container_box", "container_lid", "operator_hand", "human_body", "person"}
                for name, obj in objects.items():
                    if name in ("component_box", "red_box", "yellow_box") or (name not in EXCLUDED_NON_PAYLOAD):
                        found_target = True
                        if not obj.is_inside_container or obj.state == EntityState.EXTRACTED:
                            all_inside = False
                            break

                returned = (found_target and all_inside)
                if returned:
                    # Require that the astronaut has let go of the object to consider it fully returned
                    is_grasped = any(h.action in (HOIAction.GRASP, HOIAction.CONTACT) for h in active_hoi)
                    if not is_grasped:
                        self._accumulate_debounce(FSMStep.OBJECT_RETURNED)
                        if self.debounce_counter >= self.debounce_required:
                            self.current_step = FSMStep.OBJECT_RETURNED
                            self.candidate_step = None
                            self.debounce_counter = 0
                            self.step_start_time = now
                            self.anomaly_status = AnomalyType.NONE
                            self.anomaly_message = ""
                            self.anomaly_debounce_counter = 0
                            self.is_step_correct = True
                            self.step_verdict = "STEP OK: Nominal Procedure"
                            transition_committed = "OBJECT_RETURNED"
                    else:
                        self._reset_debounce()
                else:
                    self._reset_debounce()

            # Step 3: OBJECT_RETURNED -> Awaiting Box Close
            elif self.current_step == FSMStep.OBJECT_RETURNED:
                is_closed = (
                    lid_angle <= 28.0
                    and ("container_lid" not in objects or objects["container_lid"].bbox is None)
                )
                if is_closed:
                    self._accumulate_debounce(FSMStep.COMPLETE)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.COMPLETE
                        self.candidate_step = None
                        self.debounce_counter = self.debounce_required
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
                        transition_committed = "BOX_CLOSED"
                else:
                    self._reset_debounce()

            # Step 4: COMPLETE
            elif self.current_step == FSMStep.COMPLETE:
                self.debounce_counter = self.debounce_required
                self.is_step_correct = True
                self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
                if (now - self.step_start_time) >= 3.0:
                    is_reopening = (
                        lid_angle >= 25.0
                        or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                    )
                    if is_reopening:
                        self.current_step = FSMStep.BOX_OPENED
                        self.candidate_step = None
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "BOX_REOPENED"

        # --- GAP 5: Track highest step and detect backward regression ---
        if transition_committed:
            self.highest_committed_step = max(self.highest_committed_step, int(self.current_step))
            self.regression_debounce = 0

        if self.anomaly_status == AnomalyType.NONE:
            regression = self._check_regression(objects, lid_angle)
            if regression:
                self.anomaly_status, self.anomaly_message = regression
                self.is_step_correct = False
                self.step_verdict = f"STEP REGRESSION [{self.anomaly_status.value}]"

        # --- GAP 1: Temporal action sequence classifier ---
        if action_history and self.anomaly_status == AnomalyType.NONE:
            temporal_warning = self._check_temporal_sequence(action_history)
            if temporal_warning:
                self.anomaly_message = f"Warning: {temporal_warning}"

        # --- GAP 2: Ergonomic envelope safety check ---
        if astronaut_pose and self.anomaly_status == AnomalyType.NONE:
            ergo_result = self._check_ergonomic_safety(astronaut_pose)
            if ergo_result:
                self.anomaly_status, self.anomaly_message = ergo_result
                self.is_step_correct = False
                self.step_verdict = f"UNSAFE TECHNIQUE [{self.anomaly_status.value}]"

        # Real-time Step Correctness Verdict
        if self.anomaly_status != AnomalyType.NONE:
            self.is_step_correct = False
            self.step_verdict = f"PROCEDURAL ERROR [{self.anomaly_status.value}]"
        else:
            self.is_step_correct = True
            if self.current_step in (FSMStep.COMPLETE, FSMStep.BOX_CLOSED, FSMStep.DUAL_COMPLETE):
                self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
            else:
                self.step_verdict = "CORRECT (NOMINAL)"

        return self.current_step, self.debounce_counter, self.anomaly_status, self.anomaly_message, transition_committed

    # =====================================================================
    # Debounce Helpers (Original)
    # =====================================================================

    def _accumulate_debounce(self, target: FSMStep):
        if self.candidate_step == target:
            self.debounce_counter += 1
        else:
            self.candidate_step = target
            self.debounce_counter = 1

    def _reset_debounce(self, soft: bool = True):
        if soft and self.debounce_counter > 0:
            self.debounce_counter -= 1
            if self.debounce_counter == 0:
                self.candidate_step = None
        else:
            self.candidate_step = None
            self.debounce_counter = 0

    # =====================================================================
    # GAP 4: Config-driven forbidden_events enforcement
    # =====================================================================

    def _check_forbidden_events(
        self, objects: Dict[str, ExperimentObject], lid_angle: float, active_hoi: list
    ) -> Optional[Tuple[AnomalyType, str]]:
        """Checks config-defined forbidden_events for the current step."""
        current_step_int = int(self.current_step)
        forbidden_list = self.forbidden_events_map.get(current_step_int, [])

        for event_name in forbidden_list:
            condition_fn = FORBIDDEN_EVENT_CONDITIONS.get(event_name)
            if condition_fn and condition_fn(objects, lid_angle, active_hoi):
                # Look up alert text from config anomalies
                anomaly_info = self.config.get("anomalies", {}).get("ERROR_SEQ", {})
                alert = anomaly_info.get(
                    "alert_tts",
                    f"Warning: Forbidden action '{event_name}' detected at step {self.current_step.name}."
                )
                return AnomalyType.ERROR_SEQ, alert
        return None

    # =====================================================================
    # GAP 3: HOI event buffer for retroactive VLM validation
    # =====================================================================

    def _record_event(self, objects, lid_angle, active_hoi, frame_id):
        """Records HOI state transitions into a ring buffer for retroactive validation."""
        event = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "step": int(self.current_step),
            "lid_angle": lid_angle,
            "hoi_actions": [(h.object_name, h.action.value) for h in active_hoi],
            "object_states": {
                name: (obj.state.value, obj.is_inside_container)
                for name, obj in objects.items()
            }
        }
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.event_buffer_max:
            self.event_buffer.pop(0)

    def _retroactive_validate(self, vlm_result: Dict[str, Any]) -> Optional[Tuple[AnomalyType, str]]:
        """Scans event buffer for violations that happened between VLM queries."""
        if not self.event_buffer or not vlm_result:
            return None

        vlm_verdict = vlm_result.get("anomaly_verdict", "NOMINAL")
        if vlm_verdict != "PROCEDURAL_ERROR":
            return None

        # Scan buffer for forbidden events that fired during the buffer window
        for event in reversed(self.event_buffer):
            step_forbidden = self.forbidden_events_map.get(event["step"], [])
            for obj_name, action in event["hoi_actions"]:
                # Map HOIAction strings to config suffix strings
                suffix = action
                if action == "EXTRACT":
                    suffix = "EXTRACTED"
                elif action in ("CONTACT", "GRASP"):
                    suffix = "TOUCHED"
                
                # Build event key in the format used by forbidden_events config
                event_key = f"{obj_name.upper()}_{suffix}"
                if event_key in step_forbidden:
                    return (
                        AnomalyType.ERROR_SEQ,
                        f"Warning: Retroactive violation detected at frame {event['frame_id']}: "
                        f"{obj_name} {action} is forbidden at step {event['step']}."
                    )

        return None

    # =====================================================================
    # GAP 5: Backward regression detection
    # =====================================================================

    def _check_regression(
        self, objects: Dict[str, ExperimentObject], lid_angle: float
    ) -> Optional[Tuple[AnomalyType, str]]:
        """Detects if physical state has regressed to a prior completed step."""
        current_int = int(self.current_step)

        # Only check if we've advanced beyond step 1
        if self.highest_committed_step <= 1:
            self.regression_debounce = 0
            return None

        # Check for dual-box experiment regression scenarios
        if self.experiment_id in ("BAS-EXP-26174", "BAS-EXP-RED-YELLOW"):
            # If in step 2 (RED_EXTRACTED), returning the red box is a regression.
            if self.highest_committed_step >= 2 and current_int == 2:
                red_obj = objects.get("red_box")
                if red_obj and red_obj.is_inside_container and red_obj.state == EntityState.DOCKED:
                    self.regression_debounce += 1
                    if self.regression_debounce >= self.regression_debounce_required:
                        return (
                            AnomalyType.ERROR_REGRESSION,
                            "Warning: Wrong move! Red box has been returned to container. "
                            "This undoes a completed step. Please re-extract the red box."
                        )
                    return None

            # If in step 4 (OBJECTS_RETURNED), extracting either box is a regression.
            if self.highest_committed_step >= 4 and current_int == 4:
                red_obj = objects.get("red_box")
                yellow_obj = objects.get("yellow_box")
                
                red_extracted = red_obj and (not red_obj.is_inside_container or red_obj.state == EntityState.EXTRACTED)
                yellow_extracted = yellow_obj and (not yellow_obj.is_inside_container or yellow_obj.state == EntityState.EXTRACTED)
                
                if red_extracted or yellow_extracted:
                    self.regression_debounce += 1
                    if self.regression_debounce >= self.regression_debounce_required:
                        return (
                            AnomalyType.ERROR_REGRESSION,
                            "Warning: Wrong move! An object was extracted after both were returned. "
                            "Please leave the objects in the container and close the lid."
                        )
                    return None
        else:
            # Single-box experiment: object returned after OBJECT_EXTRACTED
            if self.highest_committed_step >= 2 and current_int == 2:
                for name, obj in objects.items():
                    if name not in ("container_box", "container_lid", "operator_hand", "human_body", "person"):
                        if obj.is_inside_container and obj.state == EntityState.DOCKED:
                            # Only flag if NOT in OBJECT_RETURNED step (step 3)
                            if current_int < 3:
                                self.regression_debounce += 1
                                if self.regression_debounce >= self.regression_debounce_required:
                                    return (
                                        AnomalyType.ERROR_REGRESSION,
                                        "Warning: Wrong move! Object returned to container prematurely. "
                                        "This undoes the extraction step."
                                    )
                                return None

        self.regression_debounce = 0
        return None

    # =====================================================================
    # GAP 1: Temporal action sequence anomaly classifier
    # =====================================================================

    def _extract_har_features(self, recent_history: List[Dict]) -> List[float]:
        """Extracts 8-dimensional feature vector from recent action history window."""
        if not recent_history:
            return [0.0] * 8

        lid_angles = [h.get("lid_angle", 0.0) for h in recent_history]
        inside_counts = [
            sum(1 for v in h.get("objects_inside", {}).values() if v)
            for h in recent_history
        ]
        total_objects = max(1, max(
            len(h.get("objects_inside", {})) for h in recent_history
        )) if recent_history else 1
        extraction_counts = [len(h.get("extracted", {})) for h in recent_history]
        hoi_counts = [len(h.get("hoi_actions", [])) for h in recent_history]

        avg_lid = sum(lid_angles) / len(lid_angles)
        is_inside_ratio = sum(inside_counts) / (len(inside_counts) * total_objects) if inside_counts else 0.0
        hand_dist = 0.15  # Default metric distance; actual dist not in history
        elbow_angle = 90.0  # Default; could be enriched later
        shoulder_angle = 60.0  # Default
        num_hoi_active = sum(hoi_counts) / len(hoi_counts) if hoi_counts else 0.0
        contact_duration = max(hoi_counts) if hoi_counts else 0
        extraction_count = max(extraction_counts) if extraction_counts else 0

        return [avg_lid, is_inside_ratio, hand_dist, elbow_angle,
                shoulder_angle, num_hoi_active, float(contact_duration),
                float(extraction_count)]

    def _check_temporal_sequence(self, action_history: List[Dict]) -> Optional[str]:
        """Uses HAR classifier to detect process-level anomalies in the action sequence."""
        if self.har_classifier is None or torch is None or len(action_history) < 10:
            return None

        recent = action_history[-10:]
        features = self._extract_har_features(recent)

        try:
            with torch.no_grad():
                input_tensor = torch.tensor([features], dtype=torch.float32)
                logits = self.har_classifier(input_tensor)
                probs = torch.softmax(logits, dim=1)
                predicted_class = probs.argmax(dim=1).item()
                confidence = probs[0, predicted_class].item()

            # If classifier confidence is very low, the action sequence is abnormal
            if confidence < self.har_anomaly_threshold:
                return (
                    f"Abnormal action sequence detected (confidence {confidence:.0%}). "
                    f"Expected step pattern not recognized."
                )

            # If predicted class doesn't match current FSM step
            expected_class = min(int(self.current_step), 4)
            if predicted_class != expected_class and confidence > 0.80:
                return (
                    f"Action pattern suggests step {predicted_class} "
                    f"but FSM is at step {int(self.current_step)}."
                )
        except Exception:
            pass  # Graceful degradation

        return None

    # =====================================================================
    # GAP 2: Wrong technique / unsafe pose detection
    # =====================================================================

    def _check_ergonomic_safety(
        self, pose: AstronautPose3D
    ) -> Optional[Tuple[AnomalyType, str]]:
        """Checks if current execution technique violates per-step ergonomic limits."""
        envelope = self.ergonomic_envelopes.get(int(self.current_step))
        if not envelope:
            return None

        violations = []
        max_elbow = envelope.get("max_elbow_angle_deg")
        if max_elbow and pose.elbow_angle_deg > max_elbow:
            violations.append(
                f"elbow angle {pose.elbow_angle_deg:.0f}° exceeds limit {max_elbow}°"
            )

        max_shoulder = envelope.get("max_shoulder_angle_deg")
        if max_shoulder and pose.shoulder_angle_deg > max_shoulder:
            violations.append(
                f"shoulder angle {pose.shoulder_angle_deg:.0f}° exceeds limit {max_shoulder}°"
            )

        if not envelope.get("rom_violation_allowed", True) and pose.rom_limits_violated:
            violations.append("range-of-motion safety limit breached")

        if violations:
            detail = "; ".join(violations)
            return (
                AnomalyType.ERROR_TECHNIQUE,
                f"Warning: Unsafe technique! {detail}. Please adjust your posture."
            )
        return None

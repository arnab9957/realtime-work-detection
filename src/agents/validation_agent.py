"""
BAS Autonomous HAR System - Agent 6: Validation Agent (Safety-Critical FSM)
Authoritative, deterministic finite state machine validating procedural compliance
with adaptive temporal debouncing, Local LLM consensus verification, and robust anomaly protection.
"""

import json
import time
from typing import Dict, Tuple, Optional, Any
from src.core.types import (
    FSMStep,
    AnomalyType,
    ExperimentObject,
    EntityState,
    HOIInteraction,
    HOIAction
)


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
        self.stall_timeout_sec: float = 60.0

        # Step Correctness Tracking (Nominal vs Anomaly)
        self.is_step_correct: bool = True
        self.step_verdict: str = "CORRECT (NOMINAL)"
        self.vlm_anomaly_explanation: Optional[str] = None

    def load_protocol(self, config_path: str):
        """Dynamically switches the active procedural FSM protocol."""
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.config_path = config_path
        self.experiment_id = self.config.get("experiment_id", "BAS-EXP-BOX-RETURN")
        self.debounce_required = self.config.get("debounce_frames", 6)
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

    def evaluate_step(
        self,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        active_hoi: list,
        current_frame: int,
        llm_verification: Optional[Dict[str, Any]] = None
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

        # Check Stall Timeout
        if self.current_step not in (FSMStep.IDLE, FSMStep.COMPLETE):
            if (now - self.step_start_time) > self.stall_timeout_sec:
                self.anomaly_status = AnomalyType.STALL_TIMEOUT
                self.anomaly_message = "Activity paused. Awaiting required procedural action."

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
            # Step 0: IDLE -> Awaiting Container Opening
            if self.current_step == FSMStep.IDLE:
                is_opening = (
                    lid_angle >= 18.0
                    or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                    or (llm_step_val is not None and llm_step_val >= 1 and llm_confidence >= 0.75 and lid_angle >= 14.0)
                    or ("red_box" in objects and objects["red_box"].bbox is not None)
                    or ("yellow_box" in objects and objects["yellow_box"].bbox is not None)
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

                # Premature close protection
                if lid_angle <= 12.0 and current_frame < 150:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 5:
                        self.anomaly_status = AnomalyType.ERROR_SKIP
                        self.anomaly_message = "Warning: Wrong move! Step skipped. Please extract the red box before closing."
                        self.is_step_correct = False
                        self.step_verdict = "PROCEDURAL ERROR [ERROR_SKIP]"
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SKIP:
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
                # Premature close protection
                if lid_angle <= 12.0 and current_frame < 320:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 5:
                        self.anomaly_status = AnomalyType.ERROR_SKIP
                        self.anomaly_message = "Warning: Wrong move! Step skipped. Please extract the yellow box before closing."
                        self.is_step_correct = False
                        self.step_verdict = "PROCEDURAL ERROR [ERROR_SKIP]"
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SKIP:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "CORRECT (NOMINAL)"

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
                # Premature close protection before return
                if lid_angle <= 12.0 and current_frame < 520:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 5:
                        self.anomaly_status = AnomalyType.ERROR_SKIP
                        self.anomaly_message = "Warning: Wrong move! Please return both boxes into the container before closing."
                        self.is_step_correct = False
                        self.step_verdict = "PROCEDURAL ERROR [ERROR_UNRETURNED_CLOSE]"
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SKIP:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "CORRECT (NOMINAL)"

                # Check both boxes returned into container
                objects_returned = False
                yellow_obj = objects.get("yellow_box")
                red_obj = objects.get("red_box")
                if (yellow_obj and yellow_obj.is_inside_container) and (red_obj and red_obj.is_inside_container):
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
                is_opening = (
                    lid_angle >= 22.0
                    or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                )
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

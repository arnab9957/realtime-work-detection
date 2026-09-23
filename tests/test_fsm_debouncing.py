"""
Unit Tests for Deterministic Finite State Machine (FSM) & 15-Frame Debouncing
ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174
Validates step transitions, temporal debouncing, and out-of-order anomaly alarms.
"""

import os
import sys
import unittest

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep, AnomalyType, EntityState, ExperimentObject, BBox2D, HOIInteraction, HOIAction, AstronautPose3D
from src.agents.validation_agent import ValidationAgent


class TestFSMDebouncing(unittest.TestCase):

    def setUp(self):
        self.validator = ValidationAgent(config_path="configs/box_return_fsm.json")
        self.deb_req = self.validator.debounce_required

    def test_nominal_step_progression_with_debounce(self):
        """Verify that state only advances after exactly debounce_required consecutive frames."""
        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
            "component_box": ExperimentObject(name="component_box", class_name="component_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Lid (deb_req frames)
        for f in range(self.deb_req - 1):
            step, deb, anomaly, _, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=f)
            self.assertEqual(step, FSMStep.IDLE, f"Step should remain IDLE on frame {f}")
            self.assertEqual(deb, f + 1)
            self.assertIsNone(trans)

        # Final frame commits the transition
        step, deb, anomaly, _, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=self.deb_req - 1)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(trans, "BOX_OPENED")
        self.assertEqual(anomaly, AnomalyType.NONE)

    def test_premature_close_anomaly(self):
        """Verify that closing lid in Step 1 before extracting object flags ERROR_SEQ instantly via config."""
        self.validator.current_step = FSMStep.BOX_OPENED
        objects = {
            "component_box": ExperimentObject(name="component_box", class_name="component_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Gap 4 makes this trigger ERROR_SEQ instantly because BOX_CLOSED is forbidden
        step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=5.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        

    def test_box_return_fsm_nominal_flow(self):
        """Validates nominal progression for the active Box Object Extraction & Return procedure."""
        val = ValidationAgent(config_path="configs/box_return_fsm.json")
        deb = val.debounce_required
        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
            "component_box": ExperimentObject(name="component_box", class_name="component_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # S0 -> S1: Box Opening
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=f)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(trans, "BOX_OPENED")

        # S1 -> S2: Object Extracted
        objects["component_box"].is_inside_container = False
        objects["component_box"].state = EntityState.EXTRACTED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=deb + f)
        self.assertEqual(step, FSMStep.OBJECT_EXTRACTED)
        self.assertEqual(trans, "OBJECT_EXTRACTED")

        # S2 -> S3: Object Returned
        objects["component_box"].is_inside_container = True
        objects["component_box"].state = EntityState.DOCKED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=2 * deb + f)
        self.assertEqual(step, FSMStep.OBJECT_RETURNED)
        self.assertEqual(trans, "OBJECT_RETURNED")

        # S3 -> S4: Box Closed
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=10.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=3 * deb + f)
        self.assertEqual(step, FSMStep.COMPLETE)
        self.assertEqual(trans, "BOX_CLOSED")
        self.assertTrue(val.is_step_correct)
        self.assertIn("NOMINAL", val.step_verdict)

    def test_isro_dual_box_nominal_flow(self):
        """Validates ISRO Benchmark PS #26174 nominal flow (Red then Yellow)."""
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-26174")
        deb = val.debounce_required

        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True),
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Container Box
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=35.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=f)
        self.assertEqual(step, FSMStep.CONTAINER_OPEN)
        self.assertEqual(trans, "CONTAINER_OPENED")
        self.assertTrue(val.is_step_correct)

        # Step 1 -> Step 2: Extract Red Box
        objects["red_box"].is_inside_container = False
        objects["red_box"].state = EntityState.EXTRACTED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=35.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=deb + f)
        self.assertEqual(step, FSMStep.RED_EXTRACTED)
        self.assertEqual(trans, "RED_BOX_EXTRACTED")
        self.assertTrue(val.is_step_correct)

        # Step 2 -> Step 3: Extract Yellow Box
        objects["yellow_box"].is_inside_container = False
        objects["yellow_box"].state = EntityState.EXTRACTED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=35.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=2 * deb + f)
        self.assertEqual(step, FSMStep.YELLOW_EXTRACTED)
        self.assertEqual(trans, "YELLOW_BOX_EXTRACTED")
        self.assertTrue(val.is_step_correct)

    def test_isro_dual_box_sequence_anomaly(self):
        """Validates that extracting Yellow Box before Red Box triggers ERROR_SEQ and marks step incorrect."""
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        deb = val.debounce_required

        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True),
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Container
        for f in range(deb):
            val.evaluate_step(objects, lid_angle=40.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=f)
        self.assertEqual(val.current_step, FSMStep.CONTAINER_OPEN)

        # Astronaut attempts to extract Yellow Box before Red Box!
        objects["yellow_box"].is_inside_container = False
        objects["yellow_box"].state = EntityState.EXTRACTED

        for f in range(5):
            step, _, anomaly, msg, _ = val.evaluate_step(objects, lid_angle=40.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=deb + f)

        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        self.assertFalse(val.is_step_correct)
        self.assertIn("PROCEDURAL ERROR", val.step_verdict)
        self.assertIn("yellow box", msg.lower())

    def test_dynamic_protocol_switching(self):
        """Verifies ValidationAgent dynamically switches protocol between Box Return and ISRO Dual-Box."""
        val = ValidationAgent(config_path="configs/box_return_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-BOX-RETURN")

        val.load_protocol("configs/experiment_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-26174")
        self.assertEqual(val.current_step, FSMStep.IDLE)
        self.assertTrue(val.is_step_correct)

        val.load_protocol("configs/box_return_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-BOX-RETURN")
        self.assertEqual(val.current_step, FSMStep.IDLE)

    def test_forbidden_events_generic_enforcement(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN
        val.highest_committed_step = 1
        objects = {
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.EXTRACTED, is_inside_container=False)
        }
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0)
        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        

    def test_per_step_stall_timeout(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN  # step 1 timeout is 30s
        import time
        val.step_start_time = time.time() - 35.0  # past 30s
        objects = {}
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0)
        self.assertEqual(anomaly, AnomalyType.STALL_TIMEOUT)
        
        val.reset()
        val.current_step = FSMStep.RED_EXTRACTED  # step 2 timeout is 45s
        val.step_start_time = time.time() - 35.0  # less than 45s
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0)
        self.assertEqual(anomaly, AnomalyType.NONE)

    def test_backward_regression_detection(self):
        val = ValidationAgent(config_path="configs/red_yellow_fsm.json")
        val.current_step = FSMStep.RED_EXTRACTED
        val.highest_committed_step = 2
        objects = {
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True)
        }
        
        for f in range(val.regression_debounce_required - 1):
            step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=f)
            self.assertEqual(anomaly, AnomalyType.NONE)
            
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=val.regression_debounce_required)
        self.assertEqual(anomaly, AnomalyType.ERROR_REGRESSION)
        self.assertIn("undoes a completed step", msg)

    def test_regression_debounce_no_false_positive(self):
        val = ValidationAgent(config_path="configs/red_yellow_fsm.json")
        val.current_step = FSMStep.RED_EXTRACTED
        val.highest_committed_step = 2
        objects = {
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True)
        }
        
        for f in range(3):
            step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=f)
            self.assertEqual(anomaly, AnomalyType.NONE)
            
        # Restore normal state
        objects["red_box"].is_inside_container = False
        objects["red_box"].state = EntityState.EXTRACTED
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=4)
        self.assertEqual(anomaly, AnomalyType.NONE)

    def test_temporal_sequence_warning(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN
        objects = {}
        # Create dummy action history
        action_history = [{"lid_angle": 60.0, "objects_inside": {}, "extracted": {}, "hoi_actions": []} for _ in range(15)]
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0, action_history=action_history)
        # We don't fail if the model is absent, but if it ran, it might add a warning
        if val.har_classifier is not None:
            self.assertTrue(msg.startswith("Warning:") or anomaly == AnomalyType.NONE)

    def test_retroactive_vlm_validation(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN
        
        # Fake an event buffer with a forbidden action
        val.event_buffer = [{
            "frame_id": 10,
            "timestamp": 12345.0,
            "step": 1,
            "lid_angle": 60.0,
            "hoi_actions": [("yellow_box", HOIAction.EXTRACT.value)],
            "object_states": {}
        }]
        llm_verif = {"anomaly_verdict": "PROCEDURAL_ERROR", "what_is_wrong": "ext"}
        objects = {}
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=15, llm_verification=llm_verif)
        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        self.assertIn("Retroactive violation", msg)

    def test_ergonomic_envelope_violation(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN  # step 1 max_elbow is 160
        objects = {}
        pose = AstronautPose3D()
        pose.elbow_angle_deg = 170.0
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0, astronaut_pose=pose)
        self.assertEqual(anomaly, AnomalyType.ERROR_TECHNIQUE)
        self.assertIn("Unsafe technique", msg)

    def test_ergonomic_envelope_no_false_positive(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN  # step 1 max_elbow is 160
        objects = {}
        pose = AstronautPose3D()
        pose.elbow_angle_deg = 120.0
        pose.shoulder_angle_deg = 100.0
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[HOIInteraction("box", "right", 0.1, HOIAction.CONTACT)], current_frame=0, astronaut_pose=pose)
        self.assertEqual(anomaly, AnomalyType.NONE)


if __name__ == "__main__":
    unittest.main()

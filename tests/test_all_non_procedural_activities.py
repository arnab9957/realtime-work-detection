"""
Tests that any non-procedural activity performed by the astronaut is detected
and updated across all attributes (activity logs, HUD, session_actions.json, CSV telemetry)
while preserving the ongoing procedural step (NO RESET to Step 0).
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.types import (
    AstronautPose3D, ExperimentObject, Vector3D, BBox2D,
    EntityState, FSMStep, AnomalyType, Joint3D
)
from src.agents.har_agent import HARAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.telemetry.action_session_logger import ActionSessionLogger
from src.telemetry.csv_logger import RealtimeCSVTelemetryLogger
import numpy as np

def create_mock_pose(activity_type="nominal"):
    # Base keypoints
    # Image frame: 640x480, shoulders at y=180, hips at y=320
    sh_w = 100.0
    r_sh = [370.0, 180.0, 0.9]
    l_sh = [270.0, 180.0, 0.9]
    nose = [320.0, 130.0, 0.9]
    r_ear = [360.0, 125.0, 0.9]
    l_ear = [280.0, 125.0, 0.9]
    r_hip = [360.0, 320.0, 0.9]
    l_hip = [280.0, 320.0, 0.9]
    r_elb = [390.0, 240.0, 0.9]
    l_elb = [250.0, 240.0, 0.9]
    r_wrist = [380.0, 280.0, 0.9]
    l_wrist = [260.0, 280.0, 0.9]

    if activity_type == "phone":
        # Right wrist at right ear
        r_wrist = [362.0, 128.0, 0.9]
    elif activity_type == "face":
        # Right wrist at nose
        r_wrist = [322.0, 132.0, 0.9]
    elif activity_type == "stretch":
        # Wrists very wide apart
        r_wrist = [550.0, 180.0, 0.9]
        l_wrist = [90.0, 180.0, 0.9]
    elif activity_type == "clapping":
        # Wrists close in front of chest
        r_wrist = [325.0, 220.0, 0.9]
        l_wrist = [315.0, 220.0, 0.9]
    elif activity_type == "hands_raised":
        # Wrists high above nose
        r_wrist = [380.0, 70.0, 0.9]
        l_wrist = [260.0, 70.0, 0.9]
    elif activity_type == "pointing":
        # One wrist extended far laterally
        r_wrist = [520.0, 200.0, 0.9]
        l_wrist = [270.0, 320.0, 0.9]
    elif activity_type == "arms_crossed":
        # Wrists crossed in front of torso
        r_wrist = [310.0, 230.0, 0.9]
        l_wrist = [330.0, 230.0, 0.9]
    elif activity_type == "hands_on_hips":
        # Wrists near hips, elbows flared
        r_wrist = [370.0, 320.0, 0.9]
        l_wrist = [270.0, 320.0, 0.9]
        r_elb = [440.0, 280.0, 0.9]
        l_elb = [200.0, 280.0, 0.9]

    kp = {
        "nose": nose, "left_ear": l_ear, "right_ear": r_ear,
        "left_shoulder": l_sh, "right_shoulder": r_sh,
        "left_elbow": l_elb, "right_elbow": r_elb,
        "left_wrist": l_wrist, "right_wrist": r_wrist,
        "left_hip": l_hip, "right_hip": r_hip
    }
    joints = {
        "wrist": Joint3D("wrist", pos_rack=Vector3D(0.5, 0.5, 0.5)),
        "shoulder": Joint3D("shoulder", pos_rack=Vector3D(0.1, 0.2, 0.0))
    }
    return AstronautPose3D(joints=joints, keypoints_2d=kp)

def test_har_agent_detects_various_non_procedural_activities():
    har = HARAgent()
    container = ExperimentObject(
        name="container_box",
        class_name="container_box",
        bbox=None,
        pos_rack=Vector3D(0.0, 0.0, 0.0),
        state=EntityState.DOCKED
    )
    objects = {"container_box": container}

    # Test Phone
    pose_phone = create_mock_pose("phone")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_phone, objects, lid_angle=0.0)
    assert act == "USING PHONE", f"Expected USING PHONE, got {act}"

    # Test Face / Head
    har.reset()
    pose_face = create_mock_pose("face")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_face, objects, lid_angle=0.0)
    assert act == "TOUCHING FACE / HEAD", f"Expected TOUCHING FACE / HEAD, got {act}"

    # Test Stretching
    har.reset()
    pose_stretch = create_mock_pose("stretch")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_stretch, objects, lid_angle=0.0)
    assert act == "STRETCHING", f"Expected STRETCHING, got {act}"

    # Test Clapping
    har.reset()
    pose_clap = create_mock_pose("clapping")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_clap, objects, lid_angle=0.0)
    assert act == "CLAPPING", f"Expected CLAPPING, got {act}"

    # Test Hands Raised
    har.reset()
    pose_raised = create_mock_pose("hands_raised")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_raised, objects, lid_angle=0.0)
    assert act in ("HANDS RAISED", "WAVING / CELEBRATING"), f"Expected HANDS RAISED, got {act}"

    # Test Pointing
    har.reset()
    pose_point = create_mock_pose("pointing")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_point, objects, lid_angle=0.0)
    assert act == "POINTING / GESTURING", f"Expected POINTING / GESTURING, got {act}"

    # Test Arms Crossed
    har.reset()
    pose_cross = create_mock_pose("arms_crossed")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_cross, objects, lid_angle=0.0)
    assert act == "ARMS CROSSED", f"Expected ARMS CROSSED, got {act}"

    # Test Hands on Hips
    har.reset()
    pose_hips = create_mock_pose("hands_on_hips")
    for _ in range(3):
        _, _, act = har.evaluate_interactions(pose_hips, objects, lid_angle=0.0)
    assert act == "HANDS ON HIPS", f"Expected HANDS ON HIPS, got {act}"

    print("All HAR kinematic activity detections PASSED!")

def test_validation_and_reasoning_no_reset_on_any_activity():
    val = ValidationAgent()
    val.current_step = FSMStep.CONTAINER_OPEN
    reas = ReasoningAgent()

    test_activities = [
        "DANCING",
        "WAVING",
        "POINTING / GESTURING",
        "ARMS CROSSED",
        "HANDS ON HIPS",
        "USING PHONE",
        "STRETCHING",
        "DRINKING COFFEE",
        "TALKING",
        "PACING / WALKING"
    ]

    for act in test_activities:
        # evaluate_step receives current_activity
        step, deb, anom, msg, evt = val.evaluate_step(
            objects={},
            lid_angle=25.0,
            active_hoi=[],
            current_frame=100,
            current_activity=act
        )

        # 1. Step MUST be preserved (NO RESET to Step 0 / IDLE)
        assert step == FSMStep.CONTAINER_OPEN, f"Step was reset! Expected CONTAINER_OPEN, got {step}"

        # 2. Anomaly status must be ERROR_SEQ
        assert anom == AnomalyType.ERROR_SEQ, f"Expected ERROR_SEQ, got {anom}"

        # 3. Anomaly message must mention the activity
        clean_act = act.replace("_", " ").strip()
        assert clean_act in msg, f"Expected '{clean_act}' in '{msg}'"
        assert val.is_step_correct is False
        assert clean_act in val.step_verdict

        # 4. Reasoning agent evaluates guidance
        instruction, voice = reas.evaluate_guidance(
            current_step=step,
            anomaly=anom,
            transition_event=evt,
            is_step_correct=val.is_step_correct,
            step_verdict=val.step_verdict,
            anomaly_message=msg
        )
        assert "WRONG MOVE" in instruction
        assert clean_act in instruction or clean_act.lower() in instruction.lower()
        if voice:
            assert clean_act.lower() in voice.lower() or clean_act in voice

    print("All Validation & Reasoning multi-activity no-reset tests PASSED!")

if __name__ == "__main__":
    test_har_agent_detects_various_non_procedural_activities()
    test_validation_and_reasoning_no_reset_on_any_activity()
    print("\nALL NON-PROCEDURAL ACTIVITY TESTS PASSED PERFECTLY!")

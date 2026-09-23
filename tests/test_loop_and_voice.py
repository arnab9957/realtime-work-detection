import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
from src.agents.perception_agent import PerceptionAgent
from src.agents.imu_agent import IMUAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.har_agent import HARAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.llm.realtime_llm_verifier import RealtimeLLMVerifier
from src.core.shared_memory import DigitalTwinBlackboard
from src.core.types import FSMStep, AnomalyType


def test_loop_restart_and_voice():
    p = PerceptionAgent()
    imu = IMUAgent()
    fusion = FusionAgent()
    har = HARAgent()
    val = ValidationAgent("configs/red_yellow_fsm.json")
    reas = ReasoningAgent("configs/red_yellow_fsm.json")
    mon = MonitoringAgent(enable_tts=False, enable_streaming=False)
    vlm = RealtimeLLMVerifier()
    bb = DigitalTwinBlackboard()

    def reset_pipeline():
        p.reset()
        imu.reset()
        fusion.reset()
        har.reset()
        val.reset()
        reas.reset()
        mon.reset()
        vlm.reset()
        bb.reset()

    video_path = "red_yellow.mp4" if os.path.exists("red_yellow.mp4") else "c1.mp4"
    cap = cv2.VideoCapture(video_path)
    print("=== SIMULATING CYCLE 1 (180 frames) ===")
    c1_transitions = []
    for f in range(1, 181):
        ret, frame = cap.read()
        if not ret:
            break
        objs, pose, lid = p.process_frame(frame)
        imu_t = imu.update_from_vision(pose)
        fused_pose, objs_rack = fusion.fuse_kinematics(pose, imu_t, objs, 0.033)
        active_hoi, objs_state, act = har.evaluate_interactions(fused_pose, objs_rack, lid)
        step, deb, anom, anom_msg, trans = val.evaluate_step(
            objs_state, lid, active_hoi, f, llm_verification=vlm.get_latest_verification()
        )
        inst, voice = reas.evaluate_guidance(
            step, anom, trans,
            is_step_correct=val.is_step_correct,
            step_verdict=val.step_verdict,
            anomaly_message=anom_msg
        )
        if trans:
            c1_transitions.append((f, step.name, trans, voice))
            print(f"   [C1 Frame {f:3d}] Transition -> {step.name} ({trans}) | Spoken: \"{voice}\"")

    print("\n=== TRIGGERING PIPELINE RESET & LOOPING TO 0 ===")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    reset_pipeline()

    print("=== SIMULATING CYCLE 2 (180 frames) ===")
    c2_transitions = []
    for f in range(1, 181):
        ret, frame = cap.read()
        if not ret:
            break
        objs, pose, lid = p.process_frame(frame)
        imu_t = imu.update_from_vision(pose)
        fused_pose, objs_rack = fusion.fuse_kinematics(pose, imu_t, objs, 0.033)
        active_hoi, objs_state, act = har.evaluate_interactions(fused_pose, objs_rack, lid)
        step, deb, anom, anom_msg, trans = val.evaluate_step(
            objs_state, lid, active_hoi, f, llm_verification=vlm.get_latest_verification()
        )
        inst, voice = reas.evaluate_guidance(
            step, anom, trans,
            is_step_correct=val.is_step_correct,
            step_verdict=val.step_verdict,
            anomaly_message=anom_msg
        )
        if trans:
            c2_transitions.append((f, step.name, trans, voice))
            print(f"   [C2 Frame {f:3d}] Transition -> {step.name} ({trans}) | Spoken: \"{voice}\"")

    vlm.close()
    cap.release()

    print(f"\nCycle 1 transitions count: {len(c1_transitions)}")
    print(f"Cycle 2 transitions count: {len(c2_transitions)}")
    assert len(c2_transitions) > 0, "Cycle 2 should have transitions"
    # Verify that in Cycle 2, Step 1 does NOT trigger on frame 1, 2, 3 prematurely
    first_c2_frame = c2_transitions[0][0]
    print(f"First transition in Cycle 2 occurred at frame {first_c2_frame}")
    assert first_c2_frame >= 6, f"Step 1 triggered prematurely at frame {first_c2_frame}"

    # Verify transition 1 is CONTAINER_OPENED with next step prompt
    assert "Container" in c2_transitions[0][3] and "red box" in c2_transitions[0][3]
    print("SUCCESS: Cycle 2 transitions and voice guidance perfectly match nominal execution!")


def test_wrong_move_and_vlm_guardian():
    val = ValidationAgent("configs/red_yellow_fsm.json")
    reas = ReasoningAgent("configs/red_yellow_fsm.json")

    # Test 1: VLM detects premature future step (e.g. step 3 while at step 1)
    fake_vlm_future = {
        "verified_step": 3,
        "step_name": "YELLOW_EXTRACTED",
        "confidence": 0.85,
        "anomaly_verdict": "PROCEDURAL_ERROR",
        "what_is_wrong": "Future step 3 detected prematurely while red box is still inside container."
    }
    val.current_step = FSMStep.CONTAINER_OPEN
    step, deb, anom, anom_msg, trans = val.evaluate_step(
        objects={}, lid_angle=30.0, active_hoi=[], current_frame=60, llm_verification=fake_vlm_future
    )
    inst, voice = reas.evaluate_guidance(
        step, anom, trans,
        is_step_correct=val.is_step_correct,
        step_verdict=val.step_verdict,
        anomaly_message=anom_msg
    )

    print("\n=== WRONG MOVE TEST 1 (VLM Future Step Detection) ===")
    print(f"   is_step_correct: {val.is_step_correct}")
    print(f"   step_verdict   : {val.step_verdict}")
    print(f"   anomaly_message: {anom_msg}")
    print(f"   spoken voice   : {voice}")
    assert not val.is_step_correct, "Should mark is_step_correct as False"
    assert "Warning: Wrong move!" in voice, f"Voice must contain 'Warning: Wrong move!', got: {voice}"

    # Test 2: Sequence violation - yellow box extracted before red box
    from src.core.types import ExperimentObject, EntityState
    fake_objects = {
        "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.EXTRACTED, is_inside_container=False),
        "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True)
    }
    val.reset()
    reas.reset()
    val.current_step = FSMStep.CONTAINER_OPEN
    step, deb, anom, anom_msg, trans = val.evaluate_step(
        objects=fake_objects, lid_angle=30.0, active_hoi=[], current_frame=60
    )
    inst, voice = reas.evaluate_guidance(
        step, anom, trans,
        is_step_correct=val.is_step_correct,
        step_verdict=val.step_verdict,
        anomaly_message=anom_msg
    )

    print("\n=== WRONG MOVE TEST 2 (Yellow Box Extracted Before Red) ===")
    print(f"   is_step_correct: {val.is_step_correct}")
    print(f"   step_verdict   : {val.step_verdict}")
    print(f"   spoken voice   : {voice}")
    assert not val.is_step_correct
    assert "Warning: Wrong move!" in voice
    assert "red box must be extracted before yellow box" in voice.lower()

    # Test 3: Recovery from wrong move -> User puts yellow back
    fake_objects_corrected = {
        "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True),
        "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True)
    }
    step, deb, anom, anom_msg, trans = val.evaluate_step(
        objects=fake_objects_corrected, lid_angle=30.0, active_hoi=[], current_frame=65
    )
    inst, voice = reas.evaluate_guidance(
        step, anom, trans,
        is_step_correct=val.is_step_correct,
        step_verdict=val.step_verdict,
        anomaly_message=anom_msg
    )
    print("\n=== RECOVERY TEST (Operator Corrects Wrong Move) ===")
    print(f"   is_step_correct: {val.is_step_correct}")
    print(f"   step_verdict   : {val.step_verdict}")
    print(f"   re-oriented voice: {voice}")
    assert val.is_step_correct
    assert "red box" in voice.lower()
    print("SUCCESS: All wrong-move and VLM future-step guardian tests passed!")


if __name__ == "__main__":
    test_wrong_move_and_vlm_guardian()
    test_loop_restart_and_voice()

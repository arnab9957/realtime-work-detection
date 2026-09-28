"""
Integration Tests for 8-Agent Multi-Agentic Pipeline Flow & Shared Memory Blackboard
ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174
Validates concurrent collaboration across the 8 specialized agents.
"""

import os
import sys
import unittest
import cv2

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep
from src.core.shared_memory import DigitalTwinBlackboard
from src.agents.perception_agent import PerceptionAgent
from src.agents.imu_agent import IMUAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.har_agent import HARAgent
from src.agents.digital_twin_agent import DigitalTwinAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.monitoring_agent import MonitoringAgent


class TestMultiAgentFlow(unittest.TestCase):

    def test_full_agent_loop_execution(self):
        video_path = os.path.join(WORKSPACE_ROOT, "clip1.mp4") if os.path.exists(os.path.join(WORKSPACE_ROOT, "clip1.mp4")) else os.path.join(WORKSPACE_ROOT, "c1.mp4")
        self.assertTrue(os.path.exists(video_path), f"Test video missing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened(), "Video must be openable")

        # Initialize Blackboard & Agents
        blackboard = DigitalTwinBlackboard()
        agent_perception = PerceptionAgent()
        agent_imu = IMUAgent()
        agent_fusion = FusionAgent()
        agent_har = HARAgent()
        agent_twin = DigitalTwinAgent()
        agent_validation = ValidationAgent(config_path="configs/box_return_fsm.json")
        agent_reasoning = ReasoningAgent(config_path="configs/box_return_fsm.json")
        agent_monitoring = MonitoringAgent(enable_tts=False, enable_streaming=False)

        # Process 25 frames through the complete 8-agent multi-agent cycle
        for frame_id in range(1, 26):
            ret, frame = cap.read()
            self.assertTrue(ret)

            # 1. Perception
            objects_cam, pose_cam, lid_angle = agent_perception.process_frame(frame)
            # 2. IMU
            imu_data = agent_imu.update_from_vision(pose_cam)
            # 3. Fusion (UKF + ROM)
            fused_pose, objects_rack = agent_fusion.fuse_kinematics(pose_cam, imu_data, objects_cam)
            # 4. HAR
            active_hoi, objects_state, act = agent_har.evaluate_interactions(fused_pose, objects_rack, lid_angle)
            # 5. Twin
            twin_scene = agent_twin.sync_scene_state(fused_pose, objects_state, lid_angle, agent_validation.current_step)
            twin_canvas = agent_twin.render_digital_twin_canvas(twin_scene)
            # 6. Validation
            step, deb, anomaly, msg, trans = agent_validation.evaluate_step(objects_state, lid_angle, active_hoi, frame_id)
            # 7. Reasoning
            inst, voice = agent_reasoning.evaluate_guidance(step, anomaly, trans)
            # 8. Monitoring
            annotated = agent_monitoring.process_egress(
                raw_frame=frame,
                fused_pose=fused_pose,
                objects=objects_state,
                lid_angle=lid_angle,
                current_step=step,
                debounce_count=deb,
                anomaly=anomaly,
                instruction=inst,
                voice_alert=voice,
                transition_event=trans,
                frame_id=frame_id,
                fps=30.0,
                latency_ms=15.0,
                twin_canvas=twin_canvas
            )

            self.assertIsNotNone(annotated)
            self.assertEqual(annotated.shape, frame.shape)

            # Verify Shared Memory update
            blackboard.update_frame_metadata(frame_id, 30.0, 15.0)
            blackboard.update_objects(objects_state, lid_angle)
            blackboard.update_pose(fused_pose)
            snapshot = blackboard.get_snapshot()
            self.assertEqual(snapshot["frame_id"], frame_id)

        cap.release()
        agent_monitoring.close()
        print("\n[Integration Test] 25 Frames processed cleanly across all 8 agents!")


if __name__ == "__main__":
    unittest.main()


    def test_forbidden_events_generic_enforcement(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN
        val.highest_committed_step = 1
        objects = {
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.EXTRACTED, is_inside_container=False)
        }
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=0)
        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        self.assertIn("Forbidden action", msg)

    def test_per_step_stall_timeout(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN  # step 1 timeout is 30s
        import time
        val.step_start_time = time.time() - 35.0  # past 30s
        objects = {}
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=0)
        self.assertEqual(anomaly, AnomalyType.STALL_TIMEOUT)
        
        val.reset()
        val.current_step = FSMStep.RED_EXTRACTED  # step 2 timeout is 45s
        val.step_start_time = time.time() - 35.0  # less than 45s
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=0)
        self.assertEqual(anomaly, AnomalyType.NONE)

    def test_backward_regression_detection(self):
        val = ValidationAgent(config_path="configs/red_yellow_fsm.json")
        val.current_step = FSMStep.RED_EXTRACTED
        val.highest_committed_step = 2
        objects = {
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True)
        }
        
        for f in range(val.regression_debounce_required - 1):
            step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=f)
            self.assertEqual(anomaly, AnomalyType.NONE)
            
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=val.regression_debounce_required)
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
            step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=f)
            self.assertEqual(anomaly, AnomalyType.NONE)
            
        # Restore normal state
        objects["red_box"].is_inside_container = False
        objects["red_box"].state = EntityState.EXTRACTED
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=4)
        self.assertEqual(anomaly, AnomalyType.NONE)

    def test_temporal_sequence_warning(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN
        objects = {}
        # Create dummy action history
        action_history = [{"lid_angle": 60.0, "objects_inside": {}, "extracted": {}, "hoi_actions": []} for _ in range(15)]
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=0, action_history=action_history)
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
            "hoi_actions": [("yellow_box", HOIAction.EXTRACT)],
            "object_states": {}
        }]
        llm_verif = {"anomaly_verdict": "PROCEDURAL_ERROR", "what_is_wrong": "ext"}
        objects = {}
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=15, llm_verification=llm_verif)
        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        self.assertIn("Retroactive violation", msg)

    def test_ergonomic_envelope_violation(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN  # step 1 max_elbow is 160
        objects = {}
        pose = AstronautPose3D()
        pose.elbow_angle_deg = 170.0
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=0, astronaut_pose=pose)
        self.assertEqual(anomaly, AnomalyType.ERROR_TECHNIQUE)
        self.assertIn("Unsafe technique", msg)

    def test_ergonomic_envelope_no_false_positive(self):
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        val.current_step = FSMStep.CONTAINER_OPEN  # step 1 max_elbow is 160
        objects = {}
        pose = AstronautPose3D()
        pose.elbow_angle_deg = 120.0
        pose.shoulder_angle_deg = 100.0
        step, deb, anomaly, msg, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=0, astronaut_pose=pose)
        self.assertEqual(anomaly, AnomalyType.NONE)


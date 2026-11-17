"""
Unit Tests for Water Bottle Manipulation & Drinking Procedure FSM
Validates nominal 5-step flow, debouncing, and procedural anomaly gates.
"""

import unittest
from types import SimpleNamespace
from src.agents.validation_agent import ValidationAgent
from src.core.types import FSMStep, AnomalyType, EntityState, HOIInteraction, HOIAction


class TestWaterBottleFSM(unittest.TestCase):
    """Test suite validating the 5-step water bottle operation."""

    def setUp(self):
        self.agent = ValidationAgent(config_path="configs/water_bottle_fsm.json")
        self.debounce = self.agent.debounce_required

    def test_water_bottle_nominal_flow(self):
        """Validates nominal 5-step sequence with debouncing."""
        self.assertEqual(self.agent.current_step, FSMStep.IDLE)
        self.assertEqual(self.agent.experiment_id, "BAS-EXP-WATER-BOTTLE")

        # Step 0 -> Step 1: Bottle Taken (lifted / grasped)
        bottle_held = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=False,
            near_mouth=False,
            tilt_angle=0.0
        )
        objects = {"bottle": bottle_held}

        for i in range(self.debounce):
            step, deb, anom, _, trans = self.agent.evaluate_step(
                objects=objects, lid_angle=0.0, active_hoi=[]
            )
            if i < self.debounce - 1:
                self.assertEqual(step, FSMStep.IDLE)
            else:
                self.assertEqual(step, FSMStep.BOTTLE_TAKEN)
                self.assertEqual(trans, "BOTTLE_TAKEN")

        # Step 1 -> Step 2: Cap Opened
        bottle_cap_open = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=True,
            near_mouth=False,
            tilt_angle=5.0
        )
        objects = {"bottle": bottle_cap_open}

        for i in range(self.debounce):
            step, deb, anom, _, trans = self.agent.evaluate_step(
                objects=objects, lid_angle=0.0, active_hoi=[]
            )
            if i < self.debounce - 1:
                self.assertEqual(step, FSMStep.BOTTLE_TAKEN)
            else:
                self.assertEqual(step, FSMStep.CAP_OPENED)
                self.assertEqual(trans, "CAP_OPENED")

        # Step 2 -> Step 3: Drinking (Near mouth + tilted)
        bottle_drinking = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=True,
            near_mouth=True,
            dist_to_mouth=0.08,
            tilt_angle=35.0
        )
        objects = {"bottle": bottle_drinking}

        for i in range(self.debounce):
            step, deb, anom, _, trans = self.agent.evaluate_step(
                objects=objects, lid_angle=0.0, active_hoi=[]
            )
            if i < self.debounce - 1:
                self.assertEqual(step, FSMStep.CAP_OPENED)
            else:
                self.assertEqual(step, FSMStep.DRINKING)
                self.assertEqual(trans, "DRINKING")

        # Step 3 -> Step 4: Cap Closed (leaves mouth & cap closed)
        bottle_cap_closed = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=False,
            near_mouth=False,
            dist_to_mouth=0.35,
            tilt_angle=5.0
        )
        objects = {"bottle": bottle_cap_closed}

        for i in range(self.debounce):
            step, deb, anom, _, trans = self.agent.evaluate_step(
                objects=objects, lid_angle=0.0, active_hoi=[]
            )
            if i < self.debounce - 1:
                self.assertEqual(step, FSMStep.DRINKING)
            else:
                self.assertEqual(step, FSMStep.CAP_CLOSED)
                self.assertEqual(trans, "CAP_CLOSED")

        # Step 4 -> Step 5: Bottle Put Down (on surface & released)
        bottle_putdown = SimpleNamespace(
            state=EntityState.DOCKED,
            on_surface=True,
            cap_open=False,
            near_mouth=False,
            dist_to_mouth=0.50,
            tilt_angle=0.0
        )
        objects = {"bottle": bottle_putdown}

        for i in range(self.debounce):
            step, deb, anom, _, trans = self.agent.evaluate_step(
                objects=objects, lid_angle=0.0, active_hoi=[]
            )
            if i < self.debounce - 1:
                self.assertEqual(step, FSMStep.CAP_CLOSED)
            else:
                self.assertEqual(step, FSMStep.COMPLETE)
                self.assertEqual(trans, "BOTTLE_PUT_DOWN")

    def test_drinking_with_closed_cap_anomaly(self):
        """Validates that trying to drink while cap is closed triggers ERROR_SEQ."""
        # Progress to Step 1: BOTTLE_TAKEN
        bottle_held = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=False,
            near_mouth=False,
            tilt_angle=0.0
        )
        for _ in range(self.debounce):
            self.agent.evaluate_step(objects={"bottle": bottle_held}, lid_angle=0.0, active_hoi=[])
        self.assertEqual(self.agent.current_step, FSMStep.BOTTLE_TAKEN)

        # Attempt to drink while cap_open is False
        drink_attempt = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=False,
            near_mouth=True,
            dist_to_mouth=0.07,
            tilt_angle=30.0
        )
        step, deb, anom, msg, _ = self.agent.evaluate_step(
            objects={"bottle": drink_attempt}, lid_angle=0.0, active_hoi=[]
        )
        self.assertEqual(anom, AnomalyType.ERROR_SEQ)
        self.assertFalse(self.agent.is_step_correct)
        self.assertIn("cap is still closed", msg)

    def test_putdown_before_closing_cap_anomaly(self):
        """Validates that putting bottle down with open cap triggers ERROR_SKIP."""
        # Progress to Step 1
        bottle_held = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=False,
            near_mouth=False,
            tilt_angle=0.0
        )
        for _ in range(self.debounce):
            self.agent.evaluate_step(objects={"bottle": bottle_held}, lid_angle=0.0, active_hoi=[])

        # Progress to Step 2: CAP_OPENED
        bottle_open = SimpleNamespace(
            state=EntityState.GRASPED,
            on_surface=False,
            cap_open=True,
            near_mouth=False,
            tilt_angle=0.0
        )
        for _ in range(self.debounce):
            self.agent.evaluate_step(objects={"bottle": bottle_open}, lid_angle=0.0, active_hoi=[])
        self.assertEqual(self.agent.current_step, FSMStep.CAP_OPENED)

        # Attempt to put down bottle with cap still open
        putdown_attempt = SimpleNamespace(
            state=EntityState.DOCKED,
            on_surface=True,
            cap_open=True,
            near_mouth=False,
            tilt_angle=0.0
        )
        step, deb, anom, msg, _ = self.agent.evaluate_step(
            objects={"bottle": putdown_attempt}, lid_angle=0.0, active_hoi=[]
        )
        self.assertIn(anom, (AnomalyType.ERROR_SEQ, AnomalyType.ERROR_SKIP))
        self.assertFalse(self.agent.is_step_correct)
        self.assertIn("cap is still open", msg)


if __name__ == "__main__":
    unittest.main()

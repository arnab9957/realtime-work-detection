import os
import sys
import unittest
import numpy as np
import joblib

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


class TestMLDistance(unittest.TestCase):
    def test_ml_distance_ridge_inference(self):
        """Test standalone ML Distance Model (34-Feature Ridge) with simulated features."""
        model_path = os.path.join(WORKSPACE_ROOT, "models", "ml_distance", "human_component_distance_model.joblib")
        self.assertTrue(os.path.exists(model_path), f"ML Model not found at: {model_path}")

        distance_model = joblib.load(model_path)
        self.assertIsNotNone(distance_model, "Model should load successfully")

        # Test simulated 34-feature vector
        features = np.zeros(34, dtype=np.float32)
        features[0] = 320   # person_centroid_x
        features[1] = 240   # person_centroid_y
        features[2] = 80    # person_centroid_z
        features[8] = 400   # chair/box_centroid_x
        features[9] = 300   # chair/box_centroid_y
        features[10] = 50   # chair/box_centroid_z
        features[17] = features[0] - features[8]
        features[18] = features[1] - features[9]
        features[19] = features[2] - features[10]
        features[23] = np.sqrt(features[17]**2 + features[18]**2 + features[19]**2)

        X_test = features.reshape(1, -1)
        predicted_distance_cm = distance_model.predict(X_test)[0]
        self.assertIsInstance(float(predicted_distance_cm), float)
        self.assertGreater(predicted_distance_cm, 0.0)


if __name__ == "__main__":
    unittest.main()

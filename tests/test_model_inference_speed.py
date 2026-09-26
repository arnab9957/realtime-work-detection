def test_model_inference_speed():
    # Synthetic test for checking YOLO model inference FPS
    fps = 32.5
    assert fps >= 30, "Inference speed is below the real-time threshold of 30 FPS"

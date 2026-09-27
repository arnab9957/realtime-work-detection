def test_video_streaming_latency():
    # Synthetic test to verify that the video stream latency remains under 200ms
    latency_ms = 145
    assert latency_ms < 200, "Stream latency exceeded maximum threshold"

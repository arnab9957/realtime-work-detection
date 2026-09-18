import json
import urllib.request
import urllib.error
import cv2
import base64
import os

def test_ollama():
    url = "http://localhost:11434/api/generate"
    print(f"Testing Ollama connection at {url}...")
    
    # 1. Simple text test to see if service is up
    test_data = {
        "model": "qwen2.5:3b",
        "prompt": "Say 'hello world' if you can read this.",
        "stream": False
    }
    
    try:
        req = urllib.request.Request(url, data=json.dumps(test_data).encode('utf-8'), headers={'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req, timeout=10)
        res_json = json.loads(response.read().decode('utf-8'))
        print("[Text Test Success] Response:", res_json.get("response", "").strip())
    except urllib.error.URLError as e:
        print(f"[Connection Error] {e}")
        print("Make sure Ollama is running and accessible at localhost:11434.")
        return
    except Exception as e:
        print(f"[Error] {e}")
        return

    # 2. Multimodal video frame test
    print("\nTesting multimodal extraction on live webcam...")
    
    def open_hardware_camera(idx: int):
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame = c.read()
                    if ret and frame is not None:
                        return c
                    c.release()
                    c = cv2.VideoCapture(idx, backend)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = c.read()
                        if ret and frame is not None:
                            return c
                    c.release()
            except Exception:
                pass
        return None
        
    cap = open_hardware_camera(0)
    if cap is None:
        print("Failed to open webcam.")
        return
    
    # Read a few frames to let the webcam auto-exposure settle
    ret = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            break
            
    cap.release()
    
    if not ret or frame is None:
        print("Failed to read frame from video.")
        return
        
    # Resize and encode frame
    frame_small = cv2.resize(frame, (640, 480))
    _, buffer = cv2.imencode('.jpg', frame_small)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    vision_data = {
        "model": "qwen3-vl:2b-instruct",
        "prompt": "You are a mission auditor. Analyze this frame. What do you see?",
        "images": [frame_b64],
        "stream": False
    }
    
    print("Sending frame to LLM (this might take a few seconds)...")
    try:
        req = urllib.request.Request(url, data=json.dumps(vision_data).encode('utf-8'), headers={'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req, timeout=30)
        res_json = json.loads(response.read().decode('utf-8'))
        
        output_text = res_json.get("response", "").strip()
        print("\n[Vision Test Success] LLM Output:")
        print(output_text)
        
        # Save output
        with open("experiments/llm_test_output.txt", "w") as f:
            f.write("--- LLM Vision Test Output ---\n")
            f.write(output_text)
        print("\nSaved output to experiments/llm_test_output.txt")
        
    except urllib.error.HTTPError as e:
        print(f"[HTTP Error] {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    test_ollama()

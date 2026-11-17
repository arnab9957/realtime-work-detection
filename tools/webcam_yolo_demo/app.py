from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    ROOT
    / "experiments/yolo26_training/bas_yolo26m_v1/weights/best.pt"
)
HTML_PATH = Path(__file__).resolve().parent / "index.html"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing to silently use CPU.")

app = FastAPI(title="BAS YOLO26 Webcam Demo")
model = YOLO(str(MODEL_PATH))

print(f"Model: {MODEL_PATH}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


@app.get("/")
async def home():
    return FileResponse(HTML_PATH)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "gpu": torch.cuda.get_device_name(0),
    }


@app.websocket("/ws")
async def webcam(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            frame_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.25,
                device=0,
                verbose=False,
            )

            annotated = results[0].plot()

            ok, encoded = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            if ok:
                await websocket.send_bytes(encoded.tobytes())

    except WebSocketDisconnect:
        print("Webcam client disconnected.")
    except Exception as exc:
        print(f"Webcam inference error: {exc}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

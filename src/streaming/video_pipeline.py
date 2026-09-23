"""
BAS Autonomous HAR System - Dual Video Pipeline
Simultaneously handles local H.264 MP4 recording and local-network IP streaming
for remote monitoring on tablet terminals and ground station consoles.
"""

import os
import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


import json


class StreamHandler(BaseHTTPRequestHandler):
    """Serves real-time MJPEG video, snapshots, and telemetry to browser clients."""

    def do_GET(self):
        if self.path in ('/stream', '/video'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            while getattr(self.server, 'running', True):
                frame_bytes = getattr(self.server, 'latest_jpeg', None)
                if frame_bytes is not None:
                    try:
                        header = (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(frame_bytes)).encode('ascii') + b'\r\n\r\n'
                        )
                        self.wfile.write(header + frame_bytes + b'\r\n')
                        self.wfile.flush()
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                        break
                time.sleep(0.033)  # ~30 FPS

        elif self.path in ('/twin_stream', '/digital_twin_stream', '/twin_video'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            while getattr(self.server, 'running', True):
                frame_bytes = getattr(self.server, 'latest_twin_jpeg', None)
                if frame_bytes is not None:
                    try:
                        header = (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(frame_bytes)).encode('ascii') + b'\r\n\r\n'
                        )
                        self.wfile.write(header + frame_bytes + b'\r\n')
                        self.wfile.flush()
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                        break
                time.sleep(0.033)  # ~30 FPS

        elif self.path.startswith('/snapshot') or self.path.startswith('/frame.jpg'):
            frame_bytes = getattr(self.server, 'latest_jpeg', None)
            if frame_bytes is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(frame_bytes)
                self.wfile.flush()
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path.startswith('/twin_snapshot') or self.path.startswith('/twin_frame.jpg'):
            frame_bytes = getattr(self.server, 'latest_twin_jpeg', None)
            if frame_bytes is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(frame_bytes)
                self.wfile.flush()
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path.startswith('/telemetry') or self.path.startswith('/api/telemetry'):
            telemetry_data = getattr(self.server, 'latest_telemetry', {})
            data_bytes = json.dumps(telemetry_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data_bytes)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            try:
                self.wfile.write(data_bytes)
                self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        elif self.path.startswith('/reset') or self.path.startswith('/api/reset'):
            # Trigger real-time experiment reset
            self.server.reset_requested = True
            resp = json.dumps({"status": "ok", "message": "Test sequence reset requested"}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(resp)
            self.wfile.flush()

        elif self.path.startswith('/api/session_actions') or self.path.startswith('/session_actions.json'):
            # Return current structured action log
            json_path = "experiments/session_actions.json"
            if os.path.exists(json_path):
                with open(json_path, "rb") as f:
                    data = f.read()
            else:
                data = json.dumps({"status": "empty", "timeline": []}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

        elif self.path.startswith('/api/digital_twin') or self.path.startswith('/digital_twin'):
            scene_graph = getattr(self.server, 'latest_scene_graph', {})
            data_bytes = json.dumps(scene_graph).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data_bytes)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(data_bytes)
            self.wfile.flush()

        elif self.path.startswith('/api/analyze') or self.path.startswith('/analyze'):
            # Trigger or poll offline LLM procedural audit (non-blocking)
            from src.llm.offline_llm_analyzer import start_async_analysis, get_latest_analysis
            latest = get_latest_analysis()
            if latest.get("status") in ("idle", "error") or "force=true" in self.path:
                start_async_analysis()
                resp_data = {"status": "running", "message": "AI Mission Audit started in background..."}
            else:
                resp_data = latest
            resp = json.dumps(resp_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(resp)
            self.wfile.flush()

        elif self.path.startswith('/api/source') or self.path.startswith('/source'):
            # Switch between live webcam and recorded demo clip
            import urllib.parse
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            target = params.get('set', [None])[0] or params.get('source', [None])[0]
            if target:
                if target in ('cam', 'webcam', '0', 'live'):
                    self.server.source_switch_requested = "0"
                elif 'red' in str(target).lower() or 'yellow' in str(target).lower():
                    self.server.source_switch_requested = "red_yellow.mp4"
                elif os.path.exists(str(target)):
                    self.server.source_switch_requested = str(target)
                else:
                    self.server.source_switch_requested = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
                resp = json.dumps({"status": "ok", "requested_source": self.server.source_switch_requested}).encode('utf-8')
            else:
                curr = getattr(self.server, 'current_source_type', 'LIVE_WEBCAM')
                resp = json.dumps({"status": "ok", "current_source": curr}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(resp)
            self.wfile.flush()

        elif self.path.startswith('/api/experiment') or self.path.startswith('/experiment'):
            # Switch between available experiment protocols
            import urllib.parse
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            exp = params.get('set', [None])[0] or params.get('exp', [None])[0]
            if exp:
                if 'dual' in exp or 'isro' in exp or '26174' in exp or 'red' in exp:
                    self.server.experiment_switch_requested = "configs/experiment_fsm.json"
                else:
                    self.server.experiment_switch_requested = "configs/box_return_fsm.json"
                resp = json.dumps({"status": "ok", "requested_protocol": self.server.experiment_switch_requested}).encode('utf-8')
            else:
                curr_exp = getattr(self.server, 'current_experiment_id', 'BAS-EXP-BOX-RETURN')
                resp = json.dumps({"status": "ok", "current_experiment": curr_exp}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(resp)
            self.wfile.flush()

        elif self.path in ('/', '/index.html'):
            # Serve Web Mission Control Dashboard
            html_path = os.path.join(os.path.dirname(__file__), "..", "gui", "web_twin", "index.html")
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-cache, must-revalidate')
                self.end_headers()
                self.wfile.write(content)
                self.wfile.flush()
            except Exception as e:
                self.send_error(404, f"Dashboard file not found: {e}")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        return


class DualVideoPipeline:
    """Coordinates local video persistence and simultaneous network IP streaming."""

    def __init__(
        self,
        local_output_path: Optional[str] = None,
        stream_port: int = 8080,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1280, 720)
    ):
        self.fps = fps
        self.resolution = resolution
        self.stream_port = stream_port
        
        # 1. Local Storage Sink
        if local_output_path is None:
            os.makedirs("experiments", exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.local_output_path = f"experiments/video_{ts}.mp4"
        else:
            self.local_output_path = local_output_path
            os.makedirs(os.path.dirname(os.path.abspath(local_output_path)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            self.local_output_path, fourcc, self.fps, self.resolution
        )

        # 2. IP Streaming Sink (HTTP MJPEG Server)
        self._server = None
        self._server_thread = None
        self._running = True
        self._start_stream_server()

    def _start_stream_server(self):
        try:
            self._server = ThreadingHTTPServer(('0.0.0.0', self.stream_port), StreamHandler)
            self._server.running = True
            self._server.latest_jpeg = None
            self._server.latest_twin_jpeg = None
            self._server.reset_requested = False
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            print(f"[Dual Video Pipeline] IP Live Streaming active at: http://localhost:{self.stream_port}/stream")
        except Exception as e:
            print(f"[Dual Video Pipeline] IP Stream Server warning: {e}")

    def check_and_clear_reset(self) -> bool:
        """Returns True if a reset was requested via /reset HTTP endpoint and clears the flag."""
        if self._server and getattr(self._server, 'reset_requested', False):
            self._server.reset_requested = False
            return True
        return False

    def check_and_clear_source_switch(self) -> Optional[str]:
        """Returns target source if a switch was requested via /api/source, and clears request."""
        if self._server and getattr(self._server, 'source_switch_requested', None):
            target = self._server.source_switch_requested
            self._server.source_switch_requested = None
            return target
        return None

    def check_and_clear_experiment_switch(self) -> Optional[str]:
        """Returns target experiment protocol path if switch was requested, and clears request."""
        if self._server and getattr(self._server, 'experiment_switch_requested', None):
            target = self._server.experiment_switch_requested
            self._server.experiment_switch_requested = None
            return target
        return None

    def set_active_source_type(self, source_type: str):
        """Updates the active source type string ('LIVE_WEBCAM' vs 'RECORDED_CLIP') on the server."""
        if self._server:
            self._server.current_source_type = source_type

    def set_active_experiment_id(self, exp_id: str):
        """Updates the active experiment identifier string on the server."""
        if self._server:
            self._server.current_experiment_id = exp_id

    def write_frame(
        self,
        frame: np.ndarray,
        telemetry: Optional[dict] = None,
        scene_graph: Optional[dict] = None,
        twin_frame: Optional[np.ndarray] = None
    ):
        """Dispatches frame to local MP4 writer and encodes JPEG for streaming clients."""
        if frame is None:
            return

        # Ensure frame matches target resolution
        h, w = frame.shape[:2]
        if (w, h) != self.resolution:
            frame_resized = cv2.resize(frame, self.resolution)
        else:
            frame_resized = frame

        # Write to local file
        if self._writer and self._writer.isOpened():
            self._writer.write(frame_resized)

        # Update streaming buffer with high-clarity encoding
        if self._server:
            ret, jpeg = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ret:
                self._server.latest_jpeg = jpeg.tobytes()
            if twin_frame is not None:
                ret_t, jpeg_t = cv2.imencode('.jpg', twin_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ret_t:
                    self._server.latest_twin_jpeg = jpeg_t.tobytes()
            if telemetry is not None:
                self._server.latest_telemetry = telemetry
            if scene_graph is not None:
                self._server.latest_scene_graph = scene_graph

    def close(self):
        self._running = False
        if self._writer:
            self._writer.release()
        if self._server:
            self._server.running = False
            self._server.shutdown()
        print(f"[Dual Video Pipeline] Recording saved to: {self.local_output_path}")

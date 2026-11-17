"""
Browser Webcam Capture
Serves a page that streams the viewer's webcam (getUserMedia) to this host as
JPEG frames, and exposes them through a cv2.VideoCapture-compatible interface
so the orchestrator can run on a remote laptop camera.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

import cv2
import numpy as np

_PAGE = """<!doctype html>
<title>BAS Browser Camera</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
video{max-width:100%;border-radius:8px;background:#000}
button{font-size:16px;padding:8px 16px;margin-right:8px}
#status{margin:12px 0;color:#9cf}
</style>
<h2>BAS Browser Camera &rarr; DGX Spark</h2>
<p>Streams this webcam into the analysis pipeline. Watch results on the
Mission Dashboard (port 8080).</p>
<button id="start">Start camera</button><button id="stop" disabled>Stop</button>
<div id="status">Idle</div>
<video id="v" autoplay muted playsinline></video>
<canvas id="c" hidden></canvas>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),ctx=c.getContext('2d'),
  st=document.getElementById('status'),bStart=document.getElementById('start'),bStop=document.getElementById('stop');
let stream=null,active=false,sent=0,t0=0;
async function loop(){
  while(active){
    ctx.drawImage(v,0,0,c.width,c.height);
    const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.8));
    try{
      await fetch('/frame',{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg'}});
      sent++;const now=performance.now();
      if(now-t0>=1000){st.textContent=`Streaming ${(sent*1000/(now-t0)).toFixed(1)} FPS`;sent=0;t0=now;}
    }catch(e){st.textContent='Upload error: '+e.message;await new Promise(r=>setTimeout(r,500));}
  }
}
bStart.onclick=async()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    v.srcObject=stream;await v.play();
    c.width=v.videoWidth;c.height=v.videoHeight;
    active=true;t0=performance.now();bStart.disabled=true;bStop.disabled=false;loop();
  }catch(e){st.textContent='Could not open camera: '+e.message+' (page must be opened via localhost or https)';}
};
bStop.onclick=()=>{active=false;if(stream)stream.getTracks().forEach(t=>t.stop());
  bStart.disabled=false;bStop.disabled=true;st.textContent='Stopped';};
</script>
"""


class BrowserCapture:
    """cv2.VideoCapture-like source fed by frames POSTed from a browser."""

    def __init__(self, port: int = 8081, wait_timeout: float = 1.0):
        self.port = port
        self.wait_timeout = wait_timeout
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._read_seq = 0
        self._cond = threading.Condition()
        self._opened = True

        capture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _cors(self):
                # The React dashboard (another origin) uploads frames here too
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self):
                body = _PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/frame":
                    self.send_response(404)
                    self._cors()
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(length)
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with capture._cond:
                        capture._frame = frame
                        capture._frame_seq += 1
                        capture._cond.notify_all()
                self.send_response(204)
                self._cors()
                self.end_headers()

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the newest unseen frame, waiting briefly for one to arrive."""
        with self._cond:
            deadline = time.time() + self.wait_timeout
            while self._frame_seq == self._read_seq:
                remaining = deadline - time.time()
                if remaining <= 0 or not self._opened:
                    return False, None
                self._cond.wait(remaining)
            self._read_seq = self._frame_seq
            return True, self._frame.copy()

    def get(self, prop_id) -> float:
        if self._frame is not None:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self._frame.shape[1])
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self._frame.shape[0])
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, prop_id, value) -> bool:
        return False

    def release(self):
        self._opened = False
        with self._cond:
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()

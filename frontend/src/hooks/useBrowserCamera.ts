import { useCallback, useEffect, useRef, useState } from 'react';
import { API } from '../services/api';

export type CameraStatus = 'off' | 'starting' | 'streaming' | 'error';

const JPEG_QUALITY = 0.8;

/**
 * Captures this browser's webcam and uploads JPEG frames to the backend's
 * browser-camera ingest. Runs inside the dashboard tab so the upload is not
 * throttled the way a hidden background tab would be.
 */
export function useBrowserCamera() {
  const [status, setStatus] = useState<CameraStatus>('off');
  const [error, setError] = useState<string | null>(null);
  const [uploadFps, setUploadFps] = useState(0);
  const streamRef = useRef<MediaStream | null>(null);
  const activeRef = useRef(false);

  const stop = useCallback(() => {
    activeRef.current = false;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setUploadFps(0);
    setStatus('off');
  }, []);

  const start = useCallback(async () => {
    if (activeRef.current) return;
    setError(null);
    setStatus('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;

      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D context unavailable');

      activeRef.current = true;
      setStatus('streaming');

      let sent = 0;
      let windowStart = performance.now();
      while (activeRef.current) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>(r => canvas.toBlob(r, 'image/jpeg', JPEG_QUALITY));
        if (!blob || !activeRef.current) continue;
        try {
          // One request in flight at a time keeps latency bounded to a single frame
          await fetch(API.CAMERA_FRAME, { method: 'POST', body: blob, headers: { 'Content-Type': 'image/jpeg' } });
          sent++;
          setStatus(prev => (prev === 'streaming' ? prev : 'streaming'));
          setError(null);
        } catch {
          setStatus('error');
          setError('Cannot reach camera ingest on port 8081');
          await new Promise(r => setTimeout(r, 500));
        }
        const now = performance.now();
        if (now - windowStart >= 1000) {
          setUploadFps(Math.round((sent * 1000) / (now - windowStart)));
          sent = 0;
          windowStart = now;
        }
      }
    } catch (err) {
      activeRef.current = false;
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      setStatus('error');
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => stop, [stop]);

  return { status, error, uploadFps, start, stop };
}

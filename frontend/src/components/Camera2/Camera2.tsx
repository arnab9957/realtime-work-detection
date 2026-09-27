import { useRef, useEffect, useState } from 'react';
import { Package } from 'lucide-react';
import { API } from '../../services/api';

// ── Camera2 panel ─────────────────────────────────────────────────────────────
export function Camera2() {
  const twinImgRef = useRef<HTMLImageElement>(null);
  const fallbackRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [streamBroken, setStreamBroken] = useState(false);

  useEffect(() => {
    const img = twinImgRef.current;
    if (!img) return;

    const onError = () => {
      setStreamBroken(true);
      if (fallbackRef.current) return;
      fallbackRef.current = setInterval(() => {
        // Double-buffered off-screen preload prevents black screen flicker
        const preloader = new Image();
        preloader.onload = () => {
          if (twinImgRef.current) twinImgRef.current.src = preloader.src;
        };
        preloader.src = `${API.TWIN_SNAPSHOT}?t=${Date.now()}`;
      }, 250);
    };

    const onLoad = () => {
      if (img.src.includes('/twin_stream')) {
        setStreamBroken(false);
        if (fallbackRef.current) {
          clearInterval(fallbackRef.current);
          fallbackRef.current = null;
        }
      }
    };

    img.addEventListener('error', onError);
    img.addEventListener('load', onLoad);
    return () => {
      img.removeEventListener('error', onError);
      img.removeEventListener('load', onLoad);
      if (fallbackRef.current) clearInterval(fallbackRef.current);
    };
  }, []);

  return (
    <div className="panel camera2-panel">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Package size={14} strokeWidth={2} style={{ color: 'var(--p-forest)' }} />
          <span className="panel-title">Camera 02 — Digital Twin Output</span>
        </div>
        <span className="panel-subtitle">
          {streamBroken ? 'SNAPSHOT MODE' : 'LIVE TWIN STREAM'}
        </span>
      </div>

      <div className="canvas-wrapper">
        <div className="stream-wrapper" style={{ width: '100%', height: '100%' }}>
          <img
            ref={twinImgRef}
            src={API.TWIN_STREAM}
            alt="Digital Twin Stream"
            className="stream-img"
          />
        </div>
      </div>
    </div>
  );
}

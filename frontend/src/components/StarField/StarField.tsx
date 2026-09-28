import { useEffect, useRef } from 'react';

interface Star {
  x: number;
  y: number;
  radius: number;
  baseAlpha: number;   // resting opacity
  alpha: number;       // current opacity
  speed: number;       // blink speed (rad/frame)
  phase: number;       // current phase offset
  color: string;
}

const STAR_COLORS = [
  'rgba(232, 238, 255,',  // cool white
  'rgba(190, 220, 255,',  // pale blue-white
  'rgba( 34, 211, 238,',  // cyan
  'rgba(165, 180, 252,',  // indigo
  'rgba(253, 230, 138,',  // warm pale gold
];

function makeStars(count: number, w: number, h: number): Star[] {
  return Array.from({ length: count }, () => {
    const radius = Math.random() < 0.85
      ? Math.random() * 0.9 + 0.2          // tiny star (majority)
      : Math.random() * 1.6 + 0.9;         // medium star
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      radius,
      baseAlpha: Math.random() * 0.55 + 0.20,
      alpha: Math.random(),
      speed: Math.random() * 0.012 + 0.003, // slow twinkle
      phase: Math.random() * Math.PI * 2,
      color: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)],
    };
  });
}

export function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const starsRef  = useRef<Star[]>([]);
  const rafRef    = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = window.innerWidth;
    let h = window.innerHeight;

    const resize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width  = w;
      canvas.height = h;
      starsRef.current = makeStars(340, w, h);
    };

    resize();
    window.addEventListener('resize', resize);

    let t = 0;
    const draw = () => {
      ctx.clearRect(0, 0, w, h);

      // Space background gradient
      const grad = ctx.createRadialGradient(w * 0.5, h * 0.35, 0, w * 0.5, h * 0.35, Math.max(w, h) * 0.85);
      grad.addColorStop(0,   '#0A0E22');
      grad.addColorStop(0.5, '#060A16');
      grad.addColorStop(1,   '#020408');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Subtle nebula smear
      const neb = ctx.createRadialGradient(w * 0.72, h * 0.25, 0, w * 0.72, h * 0.25, w * 0.38);
      neb.addColorStop(0,   'rgba(79, 70, 229, 0.055)');
      neb.addColorStop(0.5, 'rgba(34, 211, 238, 0.025)');
      neb.addColorStop(1,   'rgba(0,  0,  0,   0)');
      ctx.fillStyle = neb;
      ctx.fillRect(0, 0, w, h);

      const neb2 = ctx.createRadialGradient(w * 0.2, h * 0.7, 0, w * 0.2, h * 0.7, w * 0.32);
      neb2.addColorStop(0,   'rgba(52, 211, 153, 0.04)');
      neb2.addColorStop(1,   'rgba(0, 0, 0, 0)');
      ctx.fillStyle = neb2;
      ctx.fillRect(0, 0, w, h);

      // Stars
      t += 1;
      for (const star of starsRef.current) {
        star.phase += star.speed;
        const blink = (Math.sin(star.phase) * 0.5 + 0.5);  // 0..1
        star.alpha = star.baseAlpha * (0.45 + blink * 0.55);

        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fillStyle = `${star.color}${star.alpha.toFixed(3)})`;
        ctx.fill();

        // Soft glow for larger stars
        if (star.radius > 1.2) {
          ctx.beginPath();
          ctx.arc(star.x, star.y, star.radius * 2.8, 0, Math.PI * 2);
          const glow = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, star.radius * 2.8);
          glow.addColorStop(0,   `${star.color}${(star.alpha * 0.35).toFixed(3)})`);
          glow.addColorStop(1,   `${star.color}0)`);
          ctx.fillStyle = glow;
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="starfield-canvas" aria-hidden="true" />;
}

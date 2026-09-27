import React, { useEffect, useRef, useState } from 'react';

interface SmoothNumberOptions {
  duration?: number;
  precision?: number;
}

/**
 * Custom hook to smoothly interpolate numeric values using requestAnimationFrame.
 * Prevents abrupt jumps and flickering when telemetry arrives at high frequencies.
 */
export function useSmoothNumber(
  targetValue: number | undefined | null,
  options?: SmoothNumberOptions
): { displayValue: number | null; isChanging: boolean } {
  const duration = options?.duration ?? 280;

  const [current, setCurrent] = useState<number | null>(
    typeof targetValue === 'number' && !isNaN(targetValue) ? targetValue : null
  );
  const [isChanging, setIsChanging] = useState(false);

  const animRef = useRef<number | null>(null);
  const startValRef = useRef<number | null>(current);
  const startTimeRef = useRef<number>(0);
  const targetRef = useRef<number | null>(targetValue ?? null);
  const currentRef = useRef<number | null>(current);
  currentRef.current = current;

  useEffect(() => {
    if (targetValue === undefined || targetValue === null || isNaN(targetValue)) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      setCurrent(null);
      targetRef.current = null;
      setIsChanging(false);
      return;
    }

    if (currentRef.current === null || targetRef.current === null) {
      setCurrent(targetValue);
      startValRef.current = targetValue;
      targetRef.current = targetValue;
      return;
    }

    const prevTarget = targetRef.current;
    if (Math.abs(targetValue - prevTarget) < 0.0001) {
      return;
    }

    startValRef.current = currentRef.current;
    targetRef.current = targetValue;
    startTimeRef.current = performance.now();
    setIsChanging(true);

    const animate = (time: number) => {
      const elapsed = time - startTimeRef.current;
      const progress = Math.min(1, elapsed / duration);
      // Quintic ease-out curve for ultra-smooth landing
      const ease = 1 - Math.pow(1 - progress, 4);

      const start = startValRef.current ?? targetValue;
      const next = start + (targetValue - start) * ease;
      setCurrent(next);

      if (progress < 1) {
        animRef.current = requestAnimationFrame(animate);
      } else {
        setCurrent(targetValue);
        setIsChanging(false);
        animRef.current = null;
      }
    };

    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(animate);

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [targetValue, duration]);

  return { displayValue: current, isChanging };
}

interface SmoothNumberProps {
  value: number | undefined | null;
  precision?: number;
  prefix?: string;
  suffix?: string;
  fallback?: string;
  padLength?: number;
  className?: string;
  flash?: boolean;
}

/**
 * Flicker-free numeric display component with tabular numbers,
 * smooth interpolation, and optional subtle update highlights.
 */
export function SmoothNumber({
  value,
  precision = 0,
  prefix = '',
  suffix = '',
  fallback = '—',
  padLength,
  className = '',
  flash = false,
}: SmoothNumberProps) {
  const { displayValue, isChanging } = useSmoothNumber(value, { precision, duration: 260 });

  if (value === undefined || value === null || isNaN(value)) {
    return <span className={`smooth-num font-mono ${className}`}>{fallback}</span>;
  }

  let formatted = displayValue !== null ? displayValue.toFixed(precision) : fallback;
  if (padLength && displayValue !== null) {
    const intPart = Math.round(displayValue);
    formatted = String(intPart).padStart(padLength, '0');
  }

  return (
    <span
      className={`smooth-num font-mono tabular-nums ${isChanging && flash ? 'num-glow' : ''} ${className}`}
    >
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}

interface AnimatedTextProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  flashOnChange?: boolean;
  inline?: boolean;
}

/**
 * Animated text container that smoothly cross-fades content updates
 * and eliminates sudden text-swapping jitter.
 */
export function AnimatedText({
  children,
  className = '',
  style,
  flashOnChange = true,
  inline = false,
}: AnimatedTextProps) {
  const [currentContent, setCurrentContent] = useState(children);
  const [animClass, setAnimClass] = useState('');
  const prevRef = useRef(children);

  useEffect(() => {
    if (children !== prevRef.current) {
      prevRef.current = children;
      // Trigger a smooth CSS transition
      setAnimClass('text-fading');
      const timer = setTimeout(() => {
        setCurrentContent(children);
        setAnimClass(flashOnChange ? 'text-arrived text-flash' : 'text-arrived');
        const clearTimer = setTimeout(() => {
          setAnimClass('');
        }, 300);
        return () => clearTimeout(clearTimer);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [children, flashOnChange]);

  const Tag = inline ? 'span' : 'div';

  return (
    <Tag
      className={`smooth-text-box ${animClass} ${className}`}
      style={style}
    >
      {currentContent}
    </Tag>
  );
}

interface SmoothBadgeProps {
  label: React.ReactNode;
  variant?: 'nominal' | 'warning' | 'critical' | 'info' | 'dim';
  className?: string;
  icon?: React.ReactNode;
  pulse?: boolean;
}

/**
 * Status badge with smooth transitions for background, border, and text colors.
 */
export function SmoothBadge({
  label,
  variant = 'nominal',
  className = '',
  icon,
  pulse = false,
}: SmoothBadgeProps) {
  return (
    <span className={`smooth-badge badge-${variant} ${pulse ? 'badge-pulse' : ''} ${className}`}>
      {icon && <span className="smooth-badge-icon">{icon}</span>}
      <span className="smooth-badge-text">{label}</span>
    </span>
  );
}

"use client";
import React from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export const BackgroundBeams = React.memo(
  ({ className }: { className?: string }) => {
    // Primary beam paths based on the Aceternity curved vector matrix
    const primaryPaths = Array.from({ length: 36 }, (_, i) => {
      const x1 = -380 + i * 7;
      const y1 = -189 - i * 8;
      const x2 = -312 + i * 7;
      const y2 = 216 - i * 8;
      const x3 = 152 + i * 7;
      const y3 = 343 - i * 8;
      const x4 = 616 + i * 7;
      const y4 = 470 - i * 8;
      const x5 = 684 + i * 7;
      const y5 = 875 - i * 8;
      return `M${x1} ${y1}C${x1} ${y1} ${x2} ${y2} ${x3} ${y3}C${x4} ${y4} ${x5} ${y5} ${x5} ${y5}`;
    });

    // Secondary intersecting paths for multi-dimensional depth
    const secondaryPaths = Array.from({ length: 24 }, (_, i) => {
      const idx = i * 1.5;
      const x1 = 700 - idx * 7;
      const y1 = -200 - idx * 8;
      const x2 = 630 - idx * 7;
      const y2 = 210 - idx * 8;
      const x3 = 180 - idx * 7;
      const y3 = 350 - idx * 8;
      const x4 = -250 - idx * 7;
      const y4 = 480 - idx * 8;
      const x5 = -320 - idx * 7;
      const y5 = 880 - idx * 8;
      return `M${x1} ${y1}C${x1} ${y1} ${x2} ${y2} ${x3} ${y3}C${x4} ${y4} ${x5} ${y5} ${x5} ${y5}`;
    });

    return (
      <div
        className={cn(
          "pointer-events-none fixed inset-0 z-0 h-full w-full overflow-hidden bg-neutral-950",
          className
        )}
      >
        {/* Subtle atmospheric gradient light */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.18),rgba(255,255,255,0))]" />
        
        {/* Aceternity SVG Beams Canvas */}
        <svg
          className="absolute inset-0 h-full w-full [mask-image:radial-gradient(ellipse_at_center,transparent_15%,black_75%)]"
          viewBox="0 0 696 316"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            {/* Animated linear gradients */}
            {primaryPaths.map((_, index) => (
              <motion.linearGradient
                key={`grad-${index}`}
                id={`beam-grad-${index}`}
                gradientUnits="userSpaceOnUse"
                initial={{
                  x1: "0%",
                  x2: "0%",
                  y1: "0%",
                  y2: "0%",
                }}
                animate={{
                  x1: ["0%", "100%"],
                  x2: ["0%", "95%"],
                  y1: ["0%", "100%"],
                  y2: ["0%", "95%"],
                }}
                transition={{
                  duration: 6 + (index % 5) * 1.8,
                  repeat: Infinity,
                  repeatType: "loop",
                  ease: "linear",
                  delay: (index * 0.35) % 6,
                }}
              >
                <stop stopColor="#18CCFC" stopOpacity="0" />
                <stop stopColor="#18CCFC" stopOpacity="0.8" />
                <stop offset="32.5%" stopColor="#6344F5" stopOpacity="0.85" />
                <stop offset="100%" stopColor="#AE48FF" stopOpacity="0" />
              </motion.linearGradient>
            ))}

            {/* Radial glow filter */}
            <filter id="beam-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background subtle wire grid paths */}
          {primaryPaths.map((d, index) => (
            <path
              key={`static-p-${index}`}
              d={d}
              stroke="rgba(255, 255, 255, 0.05)"
              strokeWidth="0.8"
            />
          ))}

          {secondaryPaths.map((d, index) => (
            <path
              key={`static-s-${index}`}
              d={d}
              stroke="rgba(255, 255, 255, 0.03)"
              strokeWidth="0.6"
            />
          ))}

          {/* Glowing Animated Flowing Beams */}
          {primaryPaths.map((d, index) => (
            <motion.path
              key={`beam-${index}`}
              d={d}
              stroke={`url(#beam-grad-${index})`}
              strokeWidth={index % 3 === 0 ? "1.8" : "1.2"}
              filter="url(#beam-glow)"
              strokeLinecap="round"
              initial={{ pathLength: 0.25, pathOffset: 0, opacity: 0 }}
              animate={{
                pathOffset: [0, 1.2],
                opacity: [0, 0.85, 0.85, 0],
              }}
              transition={{
                duration: 5 + (index % 6) * 1.5,
                repeat: Infinity,
                ease: "linear",
                delay: (index * 0.4) % 5,
              }}
            />
          ))}
        </svg>

        {/* Ambient neutral-950 bottom fade to keep content grounded */}
        <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-neutral-950 to-transparent pointer-events-none" />
      </div>
    );
  }
);

BackgroundBeams.displayName = "BackgroundBeams";

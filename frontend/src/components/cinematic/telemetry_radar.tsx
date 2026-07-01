"use client";

/**
 * Telemetry Radar — Recharts Matrix Alignment Visualizer
 * ========================================================
 *
 * Renders a Radar chart mapping the 4 expert evaluation axes
 * (Psychology, Economy, Politics, Demographics) on a fixed
 * 0.0–1.0 scale using a deep dark aesthetic.
 */

import React from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface TelemetryRadarProps {
  scores: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TelemetryRadar({ scores }: TelemetryRadarProps) {
  const data = Object.entries(scores).map(([axis, value]) => ({
    axis: axis.substring(0, 4).toUpperCase(),
    fullName: axis,
    value: Math.round(value * 100) / 100,
    fullMark: 1.0,
  }));

  return (
    <div className="w-full h-64 relative">
      {/* Glow halo effect behind the chart */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-32 h-32 rounded-full bg-cyan-400/5 blur-2xl" />
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid
            stroke="#27272a"
            strokeWidth={0.5}
            gridType="polygon"
          />
          <PolarAngleAxis
            dataKey="axis"
            tick={{
              fill: "#71717a",
              fontSize: 10,
              fontFamily: "monospace",
              letterSpacing: "0.05em",
            }}
            stroke="#27272a"
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 1]}
            tick={{
              fill: "#3f3f46",
              fontSize: 8,
              fontFamily: "monospace",
            }}
            axisLine={false}
            tickCount={5}
          />
          <Radar
            name="Evaluation"
            dataKey="value"
            stroke="#22d3ee"
            strokeWidth={1.5}
            fill="#22d3ee"
            fillOpacity={0.1}
            dot={{
              r: 3,
              fill: "#22d3ee",
              stroke: "#22d3ee",
              strokeWidth: 1,
            }}
            animationDuration={800}
            animationEasing="ease-out"
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

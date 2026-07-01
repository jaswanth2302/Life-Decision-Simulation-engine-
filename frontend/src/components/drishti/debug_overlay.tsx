"use client";

import React, { useState, useEffect } from "react";
import { useDrishti } from "@/context/SimulationContext";

export function DebugOverlay() {
  const [isVisible, setIsVisible] = useState(false);
  const { debugState } = useDrishti();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Toggle on Ctrl+Shift+D or Cmd+Shift+D
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "d") {
        e.preventDefault();
        setIsVisible((v) => !v);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  if (!isVisible || !debugState) return null;

  return (
    <div
      className="fixed bottom-4 left-4 w-96 bg-black/90 text-green-400 border border-green-500/30 p-4 rounded text-xs font-mono z-50 shadow-2xl backdrop-blur-md overflow-y-auto"
      style={{ maxHeight: "80vh" }}
    >
      <div className="flex justify-between items-center mb-4 border-b border-green-500/30 pb-2">
        <h3 className="font-bold text-sm tracking-widest uppercase">
          Drishti OS Debug
        </h3>
        <button
          onClick={() => setIsVisible(false)}
          className="text-gray-500 hover:text-green-400"
        >
          ✕
        </button>
      </div>

      <div className="space-y-4">
        {/* Intent Block */}
        <section>
          <div className="text-gray-500 mb-1">Intent</div>
          <div className="font-semibold text-white">
            {debugState.intent}{" "}
            <span className="text-green-500/70">
              ({(debugState.confidence * 100).toFixed(0)}%)
            </span>
          </div>
          <div className="text-gray-400 mt-1">{debugState.reason}</div>
          <div className="text-gray-500 mt-1">
            Structural: {debugState.is_structural ? "true" : "false"}
          </div>
        </section>

        <hr className="border-green-500/20" />

        {/* Mode & Engine Block */}
        <section>
          <div className="text-gray-500 mb-1">Conversation Mode</div>
          <div className="text-white">{debugState.mode}</div>

          <div className="grid grid-cols-3 gap-2 mt-2">
            <div>
              <span className="text-gray-500">Energy</span>
              <br />
              <span className="text-white">{debugState.energy}</span>
            </div>
            <div>
              <span className="text-gray-500">Curiosity</span>
              <br />
              <span className="text-white">{debugState.curiosity}</span>
            </div>
            <div>
              <span className="text-gray-500">Certainty</span>
              <br />
              <span className="text-white">{debugState.certainty}</span>
            </div>
          </div>
        </section>

        <hr className="border-green-500/20" />

        {/* Topic Block */}
        <section>
          <div className="text-gray-500 mb-1">Topic Engine</div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-gray-500">Current</span>
              <br />
              <span className="text-white">{debugState.topic}</span>
            </div>
            <div>
              <span className="text-gray-500">Turns</span>
              <br />
              <span className="text-white">{debugState.topic_turn_count}</span>
            </div>
          </div>
        </section>

        <hr className="border-green-500/20" />

        {/* Memory & Insights Block */}
        <section>
          <div className="text-gray-500 mb-1">Recent Themes</div>
          <div className="text-white flex flex-col gap-1">
            {debugState.recent_themes && debugState.recent_themes.length > 0 ? (
              debugState.recent_themes.map((t: string, i: number) => (
                <div key={i}>- {t}</div>
              ))
            ) : (
              <span className="text-gray-600">None yet</span>
            )}
          </div>
        </section>

        <hr className="border-green-500/20" />

        {/* Summary Readiness Block */}
        <section>
          <div className="text-gray-500 mb-1">Summary Ready</div>
          <div className="text-white">
            {debugState.is_ready_for_summary ? "True" : "False"}
          </div>
          {!debugState.is_ready_for_summary && debugState.missing_information && (
            <div className="text-gray-400 mt-1 italic">
              Missing: {debugState.missing_information}
            </div>
          )}
        </section>

        <hr className="border-green-500/20" />

        {/* Latency Timing Block */}
        <section>
          <div className="text-gray-500 mb-1">Node Timing</div>
          <div className="grid grid-cols-2 gap-1 text-gray-400">
            {debugState.latencies &&
              Object.entries(debugState.latencies).map(([node, ms]) => (
                <React.Fragment key={node}>
                  <div>{node}</div>
                  <div className="text-right text-white">{ms as number} ms</div>
                </React.Fragment>
              ))}
            <div className="mt-1 pt-1 border-t border-green-500/20">Total</div>
            <div className="mt-1 pt-1 border-t border-green-500/20 text-right text-white">
              {debugState.latencies
                ? Object.values(debugState.latencies).reduce(
                    (a: any, b: any) => a + b,
                    0
                  )
                : 0}{" "}
              ms
            </div>
          </div>
        </section>

        <hr className="border-green-500/20" />

        {/* Telemetry Footer */}
        <section className="text-[10px] text-gray-600 grid grid-cols-2 gap-1 mt-4">
          <div>Build: {debugState.build}</div>
          <div>Commit: {debugState.commit}</div>
          <div>Model: {debugState.model}</div>
          <div>Prompt: {debugState.prompt_version}</div>
          <div>OS: {debugState.os_version}</div>
          <div>Session: {debugState.session_id}</div>
        </section>
      </div>
    </div>
  );
}

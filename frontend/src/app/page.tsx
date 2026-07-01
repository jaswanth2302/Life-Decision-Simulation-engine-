"use client";

import { useDrishti } from "@/context/SimulationContext";
import { LandingPage } from "@/components/drishti/landing_page";
import { IdentityForm } from "@/components/drishti/identity_form";
import { InterviewConsole } from "@/components/drishti/interview_console";
import { PersonaReveal } from "@/components/drishti/persona_reveal";
import { FeedbackPanel } from "@/components/drishti/feedback_panel";

export default function Home() {
  const { phase } = useDrishti();

  return (
    <main className="h-screen w-screen overflow-hidden select-none" style={{ background: "#080810" }}>
      {phase === "LANDING"        && <LandingPage />}
      {phase === "IDENTITY"       && <IdentityForm />}
      {phase === "INTERVIEW"      && <InterviewConsole />}
      {phase === "PERSONA_REVEAL" && <PersonaReveal />}
      {phase === "FEEDBACK"       && <FeedbackPanel />}
    </main>
  );
}

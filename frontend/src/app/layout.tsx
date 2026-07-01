import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import { DrishtiProvider } from "@/context/SimulationContext";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Drishti — Your Personal Life Simulation Engine",
  description:
    "A living model of who you are today. Not your future. Your story. Drishti builds a persistent, evolving understanding of your life through conversation.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-[#080810] text-[#f0f0f8]">
        <DrishtiProvider>{children}</DrishtiProvider>
      </body>
    </html>
  );
}

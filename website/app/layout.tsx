import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const incoming = await headers();
  const host = incoming.get("x-forwarded-host") ?? incoming.get("host") ?? "localhost:3000";
  const protocol = incoming.get("x-forwarded-proto") ?? (host.includes("localhost") ? "http" : "https");
  const base = new URL(`${protocol}://${host}`);
  const image = new URL("/og.png", base).toString();

  return {
    metadataBase: base,
    title: "Shooting Form Studio — AI Basketball Form Analysis",
    description:
      "Upload a shooting clip, detect your release frame, compare joint angles, and get practical coaching cues.",
    openGraph: {
      title: "Shooting Form Studio",
      description: "See your shot. Fix your form.",
      type: "website",
      images: [{ url: image, width: 1536, height: 1024, alt: "Shooting Form Studio" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Shooting Form Studio",
      description: "See your shot. Fix your form.",
      images: [image],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}

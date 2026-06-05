import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kaif's AI Representative",
  description: "Ask me anything about Mohd Kaif",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "VarixScan - AI Varicose Vein Detection",
  description: "AI-Powered Varicose Vein Detection and Analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
      >
        {children}
      </body>
    </html>
  );
}

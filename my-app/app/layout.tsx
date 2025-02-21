/* eslint-disable @next/next/no-page-custom-font */
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" />
        <link href="https://fonts.googleapis.com/css2?family=Bree+Serif&family=Concert+One&family=Playwrite+IT+Moderna:wght@100..400&family=Pochaevsk&display=swap" rel="stylesheet"></link>
        <link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" />
<link href="https://fonts.googleapis.com/css2?family=Germania+One&family=Lilita+One&display=swap" rel="stylesheet"></link>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" />
        <link href="https://fonts.googleapis.com/css2?family=Lilita+One&display=swap" rel="stylesheet">
          </link>

        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" />
        <link
            href="https://fonts.googleapis.com/css2?family=Bree+Serif&family=Concert+One&family=Playwrite+IT+Moderna:wght@100..400&family=Pochaevsk&display=swap"
            rel="stylesheet">
        </link>
      </head>
      <body
    
      >
        <header className="bg-white shadow-md sticky top-0 z-10">
          <div className="container mx-auto flex justify-between items-center px-6 py-4">
            <h1 className="text-2xl  tracking-wide">
              AI Wellness Coach
            </h1>
            <nav className="flex items-center space-x-6">
              <a
                href="/signup"
                className="hover:text-slate-300 transition duration-300"
              >
                Sign Up
              </a>
              <a
                href="/login"
                className="hover:text-slate-300 transition duration-300"
              >
                Log In
              </a>
              <a
                href="/get-started"
                className="bg-blue-600 text-white px-5 py-2 rounded-lg shadow-md hover:bg-blue-500 transition duration-300"
              >
                Get Started
              </a>
            </nav>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}

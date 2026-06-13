import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000'
  ),
  title: 'Abnormal Detection',
  description: 'Image manipulation detection demo using deep learning.',

  openGraph: {
    title: 'Abnormal Detection',
    description: 'Image manipulation detection demo using deep learning.',
    images: [
      {
        url: 'https://bolt.new/static/og_default.png',
        width: 1200,
        height: 630,
        alt: 'Abnormal Detection Demo',
      },
    ],
  },

  twitter: {
    card: 'summary_large_image',
    title: 'Abnormal Detection',
    description: 'Image manipulation detection demo using deep learning.',
    images: ['https://bolt.new/static/og_default.png'],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
# Tunisie Telecom Q&A Chatbot

A comprehensive Q&A chatbot for Tunisie Telecom customers, with both frontend and backend components.

[![Next.js](https://img.shields.io/badge/Next.js-13.x-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.x-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-38B2AC?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)

## Features

- Modern chat interface with Tunisie Telecom branding
- Local and web search for answering questions
- File upload support for context-aware answers
- Voice output support via ElevenLabs API 
- Fallback to LLM (Groq API) for out-of-domain questions
- Support for text, PDF, and image file uploads
- Responsive design for mobile and desktop
- Multi-language support (French, Arabic, English)
- File upload functionality
- Voice output capabilities
- Responsive design for all devices
- Graceful fallback to AI when answers aren't in the knowledge base

## Architecture

This application consists of two main components:

1. **Frontend**: Next.js application with TailwindCSS and shadcn/ui
2. **Backend**: Python FastAPI server with ML-based Q&A capabilities

## Setup and Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ with pip
- Git

### Getting Started

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd TT_Agent-main
   ```

2. Set up the backend
   ```bash
   cd backend
   ./setup.sh
   ```

3. Install frontend dependencies
   ```bash
   cd ..
   npm install
   ```

4. Start both services at once
   ```bash
   ./start.sh
   ```

5. Open your browser to http://localhost:3000

## Development

The application will now be running with both backend and frontend:

**[https://v0.dev/chat/projects/duTGJRs57yn](https://v0.dev/chat/projects/duTGJRs57yn)**

## How It Works

1. Create and modify your project using [v0.dev](https://v0.dev)
2. Deploy your chats from the v0 interface
3. Changes are automatically pushed to this repository
4. Vercel deploys the latest version from this repository

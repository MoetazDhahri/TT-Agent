#!/bin/bash
# Ollama Setup Script for TT Agent
# This script helps install and configure Ollama for use with the TT Agent

set -e  # Exit on error

echo "=== Tunisie Telecom Q&A Agent - Ollama Setup ==="
echo "This script will help you set up Ollama to provide local LLM capabilities."

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed."
else
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed successfully!"
fi

# Start Ollama service
echo "🚀 Starting Ollama service..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Give Ollama time to start up
echo "⏳ Waiting for Ollama to start..."
sleep 5

# Pull the recommended model for TT Agent
echo "📥 Downloading Llama3 model (this may take a while)..."
ollama pull llama3
echo "✅ Model downloaded successfully!"

# Check status
echo "📋 Available models:"
ollama list

echo "
=========================================================
Ollama setup complete! The service is running in the background.

• To use Ollama with the TT Agent:
  - Make sure OLLAMA_ENABLED=True in the config
  - Ollama runs by default on http://localhost:11434

• To manage Ollama:
  - List models: ollama list
  - Pull models: ollama pull <model-name>
  - Run models: ollama run <model-name>

• Recommended models for TT Agent:
  - llama3 (default)
  - mistral
  - phi
  - gemma

For more information, visit: https://ollama.ai/
=========================================================
"

# Instructions for stopping the service
echo "To stop Ollama when you're done:"
echo "  kill $OLLAMA_PID"
echo "  or run: pkill ollama"

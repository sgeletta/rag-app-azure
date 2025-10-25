#!/bin/sh

# Start the Ollama server in the background
ollama serve &

# Wait a moment for the server to start
sleep 5

# Pull the mistral model
ollama pull mistral

# Keep the script running to keep the container alive
wait
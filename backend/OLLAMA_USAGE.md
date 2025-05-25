# Ollama Integration for Tunisie Telecom Q&A System

This document explains how to use the Ollama integration in the Tunisie Telecom Q&A backend system.

## What is Ollama?

[Ollama](https://ollama.ai/) is an open-source framework for running large language models (LLMs) locally on your computer. This integration allows the TT Q&A backend to use locally hosted models as an alternative to cloud-based APIs like Grok.

## Benefits of Using Ollama

- **Privacy**: Your data stays on your machine and doesn't need to be sent to external APIs
- **Cost**: No usage fees for API calls
- **Reliability**: Works even without internet connection
- **Customization**: Choose from many different open models with different capabilities
- **Fallback**: System will automatically fall back to Grok API if Ollama is unavailable

## Setup Instructions

### 1. Install Ollama

The easiest way to install Ollama is to run the provided setup script:

```bash
cd backend
chmod +x setup_ollama.sh
./setup_ollama.sh
```

This will:
- Install Ollama if it's not already installed
- Start the Ollama service
- Download the recommended default model (llama3)

Alternatively, you can visit [ollama.ai](https://ollama.ai/) and follow their installation instructions for your platform.

### 2. Verify Installation

After installation, you can verify that Ollama is working by running:

```bash
ollama list
```

This should show the models available on your system.

### 3. Configure the Backend

The backend is already configured to use Ollama by default. You can adjust these settings in the `Config` class in `backend.py`:

```python
# Ollama configuration
OLLAMA_ENABLED = True
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # Default model to use
OLLAMA_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 1024
```

### 4. Test the Integration

You can test the Ollama integration using the provided test script:

```bash
python test_ollama.py
```

This will check if Ollama is running, list available models, and test a simple query.

## Available API Endpoints

The following API endpoints are available for managing Ollama:

### 1. Get Available Models

```
GET /ollama/models
```

Returns:
- `current_model`: The current model being used
- `available_models`: List of all available models
- `is_enabled`: Whether Ollama is enabled
- `is_available`: Whether Ollama server is reachable
- `stats`: Cache and usage statistics

### 2. Set Current Model

```
POST /ollama/model
```

Body: 
```json
"modelname"
```

Example:
```bash
curl -X POST http://localhost:8000/ollama/model -d '"mistral"'
```

### 3. Toggle Ollama

```
POST /ollama/toggle
```

Body:
```json
true
```
or
```json
false
```

Example to disable Ollama:
```bash
curl -X POST http://localhost:8000/ollama/toggle -d 'false'
```

### 4. Pull a New Model

```
POST /ollama/pull
```

Body:
```json
"modelname"
```

Example:
```bash
curl -X POST http://localhost:8000/ollama/pull -d '"phi"'
```

## Using Ollama in Queries

When making queries to the backend, you can specify whether to use Ollama:

### Text-Only Queries

```
POST /ask
```

Body:
```json
{
  "question": "What are Tunisie Telecom's data plans?",
  "language": "fr",
  "use_ollama": true
}
```

### File Upload Queries

When uploading files with queries, include the `use_ollama` form field:

```
POST /ask_with_files
```

Form data:
- `question`: Your query text
- `language`: Language code (default: "fr")
- `use_ollama`: true/false
- `files`: File uploads

## Recommended Models

Here are some recommended models to try with the TT Q&A system:

1. **llama3** (default): Good general purpose model with good performance
2. **mistral**: Fast with good capabilities for French language
3. **phi**: Smaller but efficient model
4. **gemma**: Google's model with good multilingual capabilities

You can download more models using:

```bash
ollama pull model-name
```

## Troubleshooting

1. **Ollama server not running**:
   - Start the server with `ollama serve`
   - Check if it's running on port 11434

2. **Model not found**:
   - Pull the model with `ollama pull model-name`
   - Check available models with `ollama list`

3. **Server running out of memory**:
   - Try using a smaller model like `phi` or `tinyllama`
   - Close other memory-intensive applications

4. **Slow responses**:
   - Responses depend on your hardware capabilities
   - Try models optimized for your hardware
   - Adjust `max_tokens` setting to limit response length

## Advanced Configuration

For advanced users, you can modify the Ollama client behavior in `ollama_client.py` to adjust:

- Cache size and TTL
- Timeout parameters
- Error handling behavior
- Response formatting

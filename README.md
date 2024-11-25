# Ollama Qwen API

A FastAPI wrapper for running the Ollama Qwen2.5 model locally with built-in request queuing, analytics, and error handling.

## Features

- 🚀 Simple REST API interface
- 🔄 Automatic request queuing and retry mechanism
- 📊 Optional analytics collection
- ⚡ Async processing with FastAPI
- 🛡️ Built-in error handling and recovery
- 🔍 Health monitoring endpoint

## Prerequisites

1. Install [Ollama](https://ollama.ai)
2. Pull the Qwen model:
   ```bash
   ollama pull qwen2.5:3b
   ```

## Quick Start

1. Set up your environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python api.py
   ```

The API will be available at `http://localhost:8000`

## API Reference

### Generate Text

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
    "prompt": "Your input text here",
    "max_tokens": 500,
    "temperature": 0.7,
    "store_analytics": false
}
```

**Parameters:**
- `prompt` (required): Input text for generation
- `max_tokens` (optional): Maximum tokens to generate (default: 500)
- `temperature` (optional): Creativity control (0.0-1.0, default: 0.7)
- `store_analytics` (optional): Enable analytics storage (default: false)

### Python Client Example

```python
import requests
import time
import random

def generate_text(prompt, max_retries=3, retry_delay=10):
    """
    Send request to Qwen API with automatic retry for busy server
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": random.randint(50, 600),
                    "temperature": 0.7,
                    "store_analytics": False
                }
            )
            
            if response.status_code == 503 and attempt < max_retries - 1:
                print(f"Server busy, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
                
            response.raise_for_status()
            return response.json()["generated_text"]
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None

# Usage
response = generate_text("Tell me a story about space exploration")
if response:
    print(response)
```

### Health Check

**Endpoint:** `GET /health`

Returns server and model status information.

### Curl Examples

1. Generate Text:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story about a robot",
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

2. Check Health:
```bash
curl http://localhost:8000/health
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License
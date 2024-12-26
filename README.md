# Qwen Local API Server

A FastAPI-based API server for running Qwen language models locally through Ollama with advanced features like request queuing, streaming responses, comprehensive analytics, and robust error handling.

## 🌟 Features

- **Multiple Model Support**: Run different Qwen model variants (3B, 1.5B, 0.5B)
- **Dual Response Modes**: 
  - Standard response mode for complete generations
  - Streaming mode for token-by-token output
- **Advanced Request Management**:
  - Built-in request queuing with lock-based synchronization
  - Automatic model loading and verification
  - Configurable retry mechanisms
- **Comprehensive Analytics**:
  - System performance metrics tracking
  - Optional user prompt history
  - Model usage statistics
- **Production-Ready Features**:
  - Health monitoring endpoint
  - Detailed error handling and logging
  - Configuration management
  - Resource usage tracking

## 📋 Prerequisites

1. **Ollama Installation**
   ```bash
   # For macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # For Windows
   # Download from https://ollama.ai/download
   ```

2. **Required Models**
   ```bash
   # Pull all supported models
   ollama pull qwen2.5:3b
   ollama pull qwen2.5-coder:1.5b
   ollama pull qwen2.5:0.5b
   ```

3. **Python Requirements**
   - Python 3.8+
   - FastAPI
   - Uvicorn
   - Requests
   - psutil

## 🚀 Quick Start

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # For Unix/macOS:
   source venv/bin/activate
   # For Windows:
   .\venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Edit `config.py` to customize:
     - Available models
     - Default parameters
     - Data storage settings
     - API base URL

3. **Start Server**
   ```bash
   python api.py
   ```
   Server will be available at `http://localhost:8000`

## 📖 API Reference

### Generate Text (Standard)

**Endpoint:** `POST /generate`

```json
{
    "prompt": "Your input text",
    "model": "qwen2.5:3b",
    "max_tokens": 1000,
    "temperature": 0.7,
    "store_analytics": false
}
```

**Response:**
```json
{
    "generated_text": "Model response...",
    "generation_time": 2.45,
    "model_used": "qwen2.5:3b"
}
```

### Generate Text (Streaming)

**Endpoint:** `POST /generate_stream`

- Request body same as standard generation
- Returns Server-Sent Events stream
- Each event contains a token and completion status

### Available Models

**Endpoint:** `GET /models`

Returns list of available models and their configurations.

### Health Check

**Endpoint:** `GET /health`

Returns system health status, model availability, and version information.

## 💻 Code Examples

### Python Client (Standard Generation)

```python
import requests

def generate_text(prompt, model="qwen2.5:3b", max_retries=3):
    url = "http://localhost:8000/generate"
    payload = {
        "prompt": prompt,
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["generated_text"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503 and attempt < max_retries - 1:
                print(f"Server busy, retrying... ({attempt + 1}/{max_retries})")
                continue
            raise e

# Usage
text = generate_text("Explain quantum computing")
print(text)
```

### Python Client (Streaming)

```python
import requests
import json

def stream_generation(prompt, model="qwen2.5:3b"):
    url = "http://localhost:8000/generate_stream"
    payload = {
        "prompt": prompt,
        "model": model
    }
    
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                print(data["token"], end="", flush=True)
                if data["done"]:
                    print("\nGeneration complete!")

# Usage
stream_generation("Write a story about")
```

### Curl Examples

1. **Standard Generation**
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain the theory of relativity",
       "model": "qwen2.5:3b",
       "max_tokens": 1000
     }'
   ```

2. **Streaming Generation**
   ```bash
   curl -X POST http://localhost:8000/generate_stream \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Write a poem about",
       "model": "qwen2.5:3b"
     }'
   ```

## 📊 Analytics

The server collects two types of analytics:

1. **System Metrics** (Always collected):
   - Request timestamps
   - Generation times
   - CPU/Memory usage
   - Error rates
   - Model usage statistics

2. **User Analytics** (Optional):
   - Prompt history
   - Usage patterns
   - Model preferences

Data is stored in JSON format in the `data` directory:
- `system_data.json`: System metrics
- `user_data.json`: User analytics (if enabled)

## 🔧 Configuration

Edit `config.py` to customize:

```python
MODEL_CONFIG = {
    "available_models": {
        "qwen2.5:3b": {
            "default_max_tokens": 1000,
            "default_temperature": 0.7,
        },
        # Add more models...
    },
    "default_model": "qwen2.5:0.5b",
    "api_base": "http://localhost:11434"
}

DATA_CONFIG = {
    "max_stored_requests": 5000,
    "data_directory": "data",
    "system_file": "data/system_data.json",
    "user_file": "data/user_data.json"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - See LICENSE file for details

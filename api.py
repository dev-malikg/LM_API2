# api.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import requests
import json
import logging
from functools import lru_cache
import time
from data_collector import DataCollector
from fastapi import Response
from fastapi.responses import StreamingResponse
import asyncio
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from config import MODEL_CONFIG
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LLM API", description="Local API for Ollama models")

# Add after app initialization
request_lock = Lock()

class GenerateRequest(BaseModel):
    """
    Request model for text generation
    
    Attributes:
        prompt (str): The input text to generate from
        model (str, optional): The model to use for generation
        max_tokens (int, optional): Maximum number of tokens to generate
        temperature (float, optional): Controls randomness (0.0-1.0)
        store_analytics (bool, optional): Whether to store request analytics
    """
    prompt: str
    model: Optional[str] = MODEL_CONFIG["default_model"]
    max_tokens: Optional[int] = None  # Will be set based on model
    temperature: Optional[float] = None  # Will be set based on model
    store_analytics: Optional[bool] = False

    def get_model_config(self):
        """Get configuration for the selected model"""
        model_config = MODEL_CONFIG["available_models"].get(
            self.model, 
            MODEL_CONFIG["available_models"][MODEL_CONFIG["default_model"]]
        )
        
        # Set defaults if not provided
        if self.max_tokens is None:
            self.max_tokens = model_config["default_max_tokens"]
        if self.temperature is None:
            self.temperature = model_config["default_temperature"]
        
        return model_config

class GenerateResponse(BaseModel):
    generated_text: str
    generation_time: float
    model_used: str
    
    class Config:
        protected_namespaces = ()

@lru_cache(maxsize=len(MODEL_CONFIG["available_models"]))
def check_model_loaded(model_name: str):
    """Check if specific model is loaded in Ollama."""
    try:
        response = requests.post(
            f"{MODEL_CONFIG['api_base']}/api/show",
            json={"name": model_name}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def load_model(model_name: str):
    """Load specific model into Ollama."""
    try:
        response = requests.post(
            f"{MODEL_CONFIG['api_base']}/api/pull",
            json={"name": model_name}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Ensure all models are loaded when the API starts."""
    logger.info("Starting API server...")
    for model_name in MODEL_CONFIG["available_models"].keys():
        if not check_model_loaded(model_name):
            logger.info(f"Loading model {model_name}...")
            if not load_model(model_name):
                logger.error(f"Failed to load model {model_name}")
        else:
            logger.info(f"Model {model_name} already loaded")

@app.get("/")
async def root():
    """Handle root path requests"""
    return {
        "message": "Welcome to Qwen API",
        "docs_url": "/docs",
        "available_models": list(MODEL_CONFIG["available_models"].keys()),
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "models": "/models"
        }
    }

@app.get("/models")
async def list_models():
    """List all available models and their status"""
    models = {}
    for model_name in MODEL_CONFIG["available_models"].keys():
        models[model_name] = {
            "loaded": check_model_loaded(model_name),
            "config": MODEL_CONFIG["available_models"][model_name]
        }
    return models

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the selected Qwen model."""
    
    # Validate model selection
    if request.model not in MODEL_CONFIG["available_models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model selection. Available models: {list(MODEL_CONFIG['available_models'].keys())}"
        )
    
    # Get model-specific configuration
    request.get_model_config()
    
    # Try to acquire the lock
    if not request_lock.acquire(blocking=False):
        logger.info("Server busy - rejected incoming request")
        raise HTTPException(
            status_code=503,
            detail="Server is currently processing another request. Please try again in a few seconds."
        )
    
    logger.info(f"Received generation request for model {request.model} - prompt length: {len(request.prompt)} chars")
    try:
        # Start timing
        start_time = time.time()

        # Prepare the request to Ollama
        ollama_request = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature
            }
        }

        # Make request to Ollama
        response = requests.post(
            f"{MODEL_CONFIG['api_base']}/api/generate",
            json=ollama_request
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to generate text from model"
            )

        # Calculate elapsed time
        generation_time = time.time() - start_time

        # Parse the response
        result = response.json()

        # Log successful response
        logger.info(f"Request completed - generation time: {generation_time:.2f}s")
        return GenerateResponse(
            generated_text=result.get("response", ""),
            generation_time=round(generation_time, 2),
            model_used=request.model
        )

    except requests.exceptions.ConnectionError:
        logger.error("Connection to Ollama failed")
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Please check if Ollama is running on your machine."
        )
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong during text generation. Please check the logs for details."
        )
    finally:
        # Always release the lock
        request_lock.release()


@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    """Generate text using the selected Qwen model with streaming response."""
    
    # Validate model selection
    if request.model not in MODEL_CONFIG["available_models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model selection. Available models: {list(MODEL_CONFIG['available_models'].keys())}"
        )
    
    # Get model-specific configuration
    request.get_model_config()
    
    # Try to acquire the lock
    if not request_lock.acquire(blocking=False):
        logger.info("Server busy - rejected incoming request")
        raise HTTPException(
            status_code=503,
            detail="Server is currently processing another request. Please try again in a few seconds."
        )
    
    logger.info(f"Received streaming request for model {request.model} - prompt length: {len(request.prompt)} chars")
    
    async def generate():
        try:
            start_time = time.time()
            
            # Prepare the request to Ollama
            ollama_request = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": True,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature
                }
            }
            
            # Make streaming request to Ollama
            with requests.post(
                f"{MODEL_CONFIG['api_base']}/api/generate",
                json=ollama_request,
                stream=True
            ) as response:
                
                if response.status_code != 200:
                    error_msg = json.dumps({
                        "error": "Failed to generate text from model",
                        "status_code": response.status_code
                    }) + "\n"
                    yield error_msg.encode('utf-8')
                    return
                
                total_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        total_response += token
                        # Send each token as a JSON object
                        yield json.dumps({
                            "token": token,
                            "done": data.get("done", False)
                        }).encode('utf-8') + b"\n"
                
                # Log metrics after completion
                generation_time = time.time() - start_time
                asyncio.create_task(log_request(
                    prompt=request.prompt,
                    success=True,
                    response_time=generation_time,
                    response_text=total_response,
                    store_analytics=request.store_analytics,
                    model=request.model
                ))
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            error_msg = json.dumps({
                "error": str(e),
                "status_code": 500
            }) + "\n"
            yield error_msg.encode('utf-8')
            
            # Log error metrics
            asyncio.create_task(log_request(
                prompt=request.prompt,
                success=False,
                response_time=0,
                response_text=None,
                error=str(e),
                model=request.model
            ))
            
        finally:
            # Always release the lock
            request_lock.release()
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    model_statuses = {
        model_name: check_model_loaded(model_name)
        for model_name in MODEL_CONFIG["available_models"].keys()
    }
    all_models_loaded = all(model_statuses.values())
    return {
        "status": "healthy" if all_models_loaded else "degraded",
        "model_status": model_statuses,
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# Initialize data collector
data_collector = DataCollector()

# Add this middleware before the routes
@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    if request.url.path == "/generate":
        start_time = time.time()
        
        try:
            # Get request body
            body = await request.json()
            prompt = body.get("prompt", "")
            store_analytics = body.get("store_analytics", False)
            model = body.get("model", MODEL_CONFIG["default_model"])
            
            # Process the request
            response = await call_next(request)
            
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Parse response
            response_data = json.loads(response_body)
            generation_time = response_data.get("generation_time", 0)
            generated_text = response_data.get("generated_text", "")
            
            # Log metrics in background tasks
            asyncio.create_task(log_request(
                prompt=prompt,
                success=True,
                response_time=generation_time,
                response_text=generated_text,
                store_analytics=store_analytics,
                model=model
            ))
            
            # Return response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {str(e)}")
            asyncio.create_task(log_request(
                prompt=body.get("prompt", "") if 'body' in locals() else "",
                success=False,
                response_time=0,
                response_text=None,
                error=str(e),
                model=body.get("model", MODEL_CONFIG["default_model"]) if 'body' in locals() else MODEL_CONFIG["default_model"]
            ))
            return await call_next(request)
    
    return await call_next(request)

async def log_request(prompt: str, success: bool, response_time: float, response_text: str = None, 
                     store_analytics: bool = False, error: str = None, model: str = None):
    """Background task for logging requests"""
    try:
        # Log system metrics
        data_collector.log_system_metrics(
            prompt=prompt,
            response=response_text,
            generation_time=response_time,
            error={"type": type(error).__name__, "message": str(error)} if error else None,
            model=model
        )
        
        # Log user prompt if analytics enabled
        if store_analytics:
            await data_collector.log_user_prompt(prompt, model)
            
    except Exception as e:
        logger.error(f"Error logging request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
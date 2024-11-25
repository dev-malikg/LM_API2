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
app = FastAPI(title="Qwen API", description="Local API for Ollama Qwen2.5 model")

# Add after app initialization
request_lock = Lock()

class GenerateRequest(BaseModel):
    """
    Request model for text generation
    
    Attributes:
        prompt (str): The input text to generate from
        max_tokens (int, optional): Maximum number of tokens to generate
        temperature (float, optional): Controls randomness (0.0-1.0)
        store_analytics (bool, optional): Whether to store request analytics
    """
    prompt: str
    max_tokens: Optional[int] = MODEL_CONFIG["default_max_tokens"]
    temperature: Optional[float] = MODEL_CONFIG["default_temperature"]
    store_analytics: Optional[bool] = False

class GenerateResponse(BaseModel):
    generated_text: str
    generation_time: float

@lru_cache(maxsize=1)
def check_model_loaded():
    """Check if the model is loaded in Ollama."""
    try:
        response = requests.post(
            f"{MODEL_CONFIG['api_base']}/api/show",
            json={"name": MODEL_CONFIG["model"]}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def load_model():
    """Load the model into Ollama."""
    try:
        response = requests.post(
            f"{MODEL_CONFIG['api_base']}/api/pull",
            json={"name": MODEL_CONFIG["model"]}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Ensure model is loaded when the API starts."""
    logger.info("Starting API server...")
    if not check_model_loaded():
        logger.info("Model not loaded. Loading model...")
        if not load_model():
            logger.error("Failed to load model. Please ensure Ollama is running.")
            # Continue running the server even if model loading fails
    else:
        logger.info("Model already loaded")

@app.get("/")
async def root():
    """Handle root path requests"""
    return {
        "message": "Welcome to Qwen API",
        "docs_url": "/docs",  # FastAPI's automatic API documentation
        "endpoints": {
            "generate": "/generate",
            "health": "/health"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the Qwen model."""
    
    # Try to acquire the lock
    if not request_lock.acquire(blocking=False):
        logger.info("Server busy - rejected incoming request")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Server is currently processing another request. Please try again in a few seconds."
        )
    
    logger.info(f"Received generation request - prompt length: {len(request.prompt)} chars")
    try:
        # Start timing
        start_time = time.time()

        # Prepare the request to Ollama
        ollama_request = {
            "model": MODEL_CONFIG["model"],
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
            generation_time=round(generation_time, 2)
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

@app.get("/health")
async def health_check():
    model_status = check_model_loaded()
    return {
        "status": "healthy" if model_status else "degraded",
        "model_loaded": model_status,
        "timestamp": time.time(),
        "version": "1.0.0"  # Add version tracking
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
                store_analytics=store_analytics
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
                error=str(e)
            ))
            return await call_next(request)
    
    return await call_next(request)

async def log_request(prompt: str, success: bool, response_time: float, response_text: str = None, store_analytics: bool = False, error: str = None):
    """Background task for logging requests"""
    try:
        # Log system metrics
        data_collector.log_system_metrics(
            prompt=prompt,
            response=response_text,
            generation_time=response_time,
            error={"type": type(error).__name__, "message": str(error)} if error else None
        )
        
        # Log user prompt if analytics enabled
        if store_analytics:
            await data_collector.log_user_prompt(prompt)
            
    except Exception as e:
        logger.error(f"Error logging request: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 
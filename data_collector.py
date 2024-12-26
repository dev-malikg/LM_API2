# data_collector.py

import json
import time
from datetime import datetime
import os
from threading import Lock
import logging
import psutil
from config import DATA_CONFIG

class DataCollector:
    def __init__(self):
        """Initialize the DataCollector with necessary files and locks"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_CONFIG["data_directory"], exist_ok=True)
        
        # Initialize file paths from config
        self.system_file = DATA_CONFIG["system_file"]
        self.user_file = DATA_CONFIG["user_file"]
        
        # Thread locks for safe file access
        self.system_lock = Lock()
        self.user_lock = Lock()
        
        # Create initial files if needed
        self._init_files()
    
    def _init_files(self):
        """Create initial JSON files with empty data structure if they don't exist"""
        system_data = {
            "requests": [],
            "total_requests": 0,
            "errors": 0,
            "model_usage": {}  # Track usage per model
        }
        
        user_data = {
            "prompts": [],
            "total_prompts": 0
        }
        
        # Create files if they don't exist
        if not os.path.exists(self.system_file):
            with open(self.system_file, 'w') as f:
                json.dump(system_data, f, indent=2)
                
        if not os.path.exists(self.user_file):
            with open(self.user_file, 'w') as f:
                json.dump(user_data, f, indent=2)
        
        try:
            # Test write permissions
            if os.path.exists(self.system_file) and not os.access(self.system_file, os.W_OK):
                logging.error("No write permission for system data file")
            if os.path.exists(self.user_file) and not os.access(self.user_file, os.W_OK):
                logging.error("No write permission for user data file")
                
        except Exception as e:
            logging.error(f"Error initializing data files: {e}")

    def _detect_prompt_type(self, prompt: str) -> str:
        """Detect the type of prompt based on content"""
        # [Previous implementation remains the same]
        # ... [Keep existing implementation]

    def _get_system_usage(self):
        """Get current CPU and memory usage"""
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }

    def _format_timestamp(self, timestamp):
        """Convert timestamp to readable format"""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    # SECTION 1: System Metrics Collection
    def log_system_metrics(self, prompt: str, response: str = None, generation_time: float = None, 
                         error: dict = None, model: str = None):
        """Log detailed system performance metrics"""
        with self.system_lock:
            try:
                if error:
                    logging.info("Logging failed request metrics")
                
                with open(self.system_file, 'r') as f:
                    data = json.load(f)
                
                # Get system usage
                system_usage = self._get_system_usage()
                
                # Create detailed request data
                request = {
                    "timestamp": self._format_timestamp(time.time()),
                    "prompt_length": len(prompt),
                    "prompt_type": self._detect_prompt_type(prompt),
                    "response_tokens": len(response.split()) if response else 0,
                    "generation_time": round(generation_time, 2) if generation_time else 0.0,
                    "cpu_usage": round(system_usage["cpu"], 1),
                    "memory_usage": round(system_usage["memory"], 2),
                    "model": model,  # Add model information
                    "error_type": error.get("type") if error else None,
                    "error_message": error.get("message") if error else None
                }
                
                # Update model usage statistics
                if model:
                    if model not in data.get("model_usage", {}):
                        data["model_usage"][model] = 0
                    data["model_usage"][model] += 1
                
                data["requests"].append(request)
                data["total_requests"] += 1
                if error:
                    data["errors"] += 1
                
                # Keep last requests according to config
                data["requests"] = data["requests"][-DATA_CONFIG["max_stored_requests"]:]
                
                with open(self.system_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logging.info("Successfully logged system metrics")
                    
            except Exception as e:
                logging.error(f"Failed to log system metrics: {e}")

    # SECTION 2: User History Collection
    async def log_user_prompt(self, prompt: str, model: str = None):
        """Log user prompt history"""
        with self.user_lock:
            try:
                with open(self.user_file, 'r') as f:
                    data = json.load(f)
                
                # Add new prompt with formatted timestamp and model info
                prompt_data = {
                    "timestamp": self._format_timestamp(time.time()),
                    "prompt": prompt,
                    "length": len(prompt),
                    "model": model  # Add model information
                }
                
                data["prompts"].append(prompt_data)
                data["total_prompts"] += 1
                
                # Keep last 1000 prompts only
                data["prompts"] = data["prompts"][-1000:]
                
                with open(self.user_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                logging.error(f"Failed to log user prompt: {e}")

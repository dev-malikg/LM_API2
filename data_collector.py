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
            "errors": 0
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
        prompt_lower = prompt.lower()
        
        # Define keyword categories
        keywords = {
            "creative": ["write", "create", "generate", "design", "compose", "draft", "make"],
            "question": ["?", "who", "when", "where", "why"],
            "explanation": ["explain", "describe", "how", "what", "tell me about", "elaborate"],
            "comparison": ["compare", "difference", "versus", "vs", "better", "advantages", "disadvantages"],
            "analysis": ["analyze", "examine", "evaluate", "assess", "review"],
            "troubleshooting": ["debug", "fix", "solve", "issue", "error", "problem", "wrong"],
            "implementation": ["implement", "code", "program", "develop", "build"],
            "optimization": ["optimize", "improve", "enhance", "speed up", "efficient"],
            "validation": ["check", "verify", "validate", "test", "confirm"]
        }
        
        # Check each category
        for category, words in keywords.items():
            if any(word in prompt_lower for word in words):
                return category
                
        # Check for code-specific patterns
        if any(lang in prompt_lower for lang in ["python", "javascript", "java", "cpp", "ruby", "sql"]):
            return "code_related"
            
        # Check for list/enumeration requests
        if any(pattern in prompt_lower for pattern in ["list", "enumerate", "what are", "examples of"]):
            return "enumeration"
            

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
    def log_system_metrics(self, prompt: str, response: str = None, generation_time: float = None, error: dict = None):
        """Log detailed system performance metrics"""
        with self.system_lock:
            try:
                # Only log start of metrics collection if there's an error
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
                    "error_type": error.get("type") if error else None,
                    "error_message": error.get("message") if error else None
                }
                
                data["requests"].append(request)
                data["total_requests"] += 1
                if error:
                    data["errors"] += 1
                
                # Keep last 1000 requests only
                data["requests"] = data["requests"][-DATA_CONFIG["max_stored_requests"]:]
                
                with open(self.system_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logging.info("Successfully logged system metrics")
                    
            except Exception as e:
                logging.error(f"Failed to log system metrics: {e}")

    # SECTION 2: User History Collection
    async def log_user_prompt(self, prompt: str):
        """Log user prompt history"""
        with self.user_lock:
            try:
                with open(self.user_file, 'r') as f:
                    data = json.load(f)
                
                # Add new prompt with formatted timestamp
                prompt_data = {
                    "timestamp": self._format_timestamp(time.time()),  # Use the same formatting method
                    "prompt": prompt,
                    "length": len(prompt)
                }
                
                data["prompts"].append(prompt_data)
                data["total_prompts"] += 1
                
                # Keep last 1000 prompts only
                data["prompts"] = data["prompts"][-1000:]
                
                with open(self.user_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                logging.error(f"Failed to log user prompt: {e}")


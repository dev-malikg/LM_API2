# Basic configuration for the Qwen API
MODEL_CONFIG = {
    "model": "qwen2.5:3b",
    "default_max_tokens": 1000,
    "default_temperature": 0.7,
    "api_base": "http://localhost:11434"
}

# Data storage configuration
DATA_CONFIG = {
    "max_stored_requests": 5000,  # Maximum number of requests to store
    "data_directory": "data",     # Directory for storing data files
    "system_file": "data/system_data.json",
    "user_file": "data/user_data.json"
} 
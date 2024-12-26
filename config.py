# config.py

# Basic configuration for the Qwen API
MODEL_CONFIG = {
    "available_models": {
        "qwen2.5:3b": {
            "default_max_tokens": 1000,
            "default_temperature": 0.7,
        },
        "qwen2.5-coder:1.5b": {
            "default_max_tokens": 1000,
            "default_temperature": 0.1,
        },
        "qwen2.5:0.5b": {
            "default_max_tokens": 800,  # Lower for smaller model
            "default_temperature": 0.7,
        }
        # add mode models you want ...
    },
    "default_model": "qwen2.5:0.5b",  # Set default model
    "api_base": "http://localhost:11434"
}

# Data storage configuration
DATA_CONFIG = {
    "max_stored_requests": 5000,  # Maximum number of requests to store
    "data_directory": "data",     # Directory for storing data files
    "system_file": "data/system_data.json",
    "user_file": "data/user_data.json"
}
from typing import Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    def __init__(self, name: str, base_url: str, model_name: str, api_key_env: str):
        self.name = name
        self.base_url = base_url
        self.model_name = model_name
        self.api_key_env = api_key_env

# Define all supported models
SUPPORTED_MODELS = {
    "gemini": ModelConfig(
        name="Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY"
    ),
    "claude": ModelConfig(
        name="Claude",
        base_url="https://api.anthropic.com/v1/",
        model_name="claude-3-5-sonnet-20240620",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    "llama": ModelConfig(
        name="Llama",
        base_url="https://api.llama.com/compat/v1/",
        model_name="Llama-3.3-8B-Instruct",
        api_key_env="LLAMA_API_KEY"
    ),
    "deepseek": ModelConfig(
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        model_name="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY"
    ),
    "openai": ModelConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1/",
        model_name="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY"
    )
}

def get_available_models() -> Dict[str, ModelConfig]:
    """Returns a dictionary of model names to their configs for models that have API keys set."""
    return {
        name: config for name, config in SUPPORTED_MODELS.items()
        if os.getenv(config.api_key_env)
    }

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Returns the configuration for a specific model if it's available."""
    if model_name in SUPPORTED_MODELS:
        config = SUPPORTED_MODELS[model_name]
        if os.getenv(config.api_key_env):
            return config
    return None
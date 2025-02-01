"""
Configuration settings for the LLM Factory module.

This module defines the configuration settings for the LLM Factory module,
including settings for various LLM providers such as OpenAI, Ollama, and Groq.
It uses pydantic_settings for type-safe configuration management and
environment variable loading.

Classes:
    LLMProviderSettings: Base settings for LLM providers.
    OpenAISettings: Settings specific to OpenAI.
    OllamaSettings: Settings specific to Ollama.
    GroqSettings: Settings specific to Groq.
    AnthropicSettings: Settings specific to Anthropic.
    HuggingFaceSettings: Settings for Hugging Face models.
    Settings: Main settings class for the application.

Functions:
    get_settings: Cached function to retrieve application settings.

Usage:
    from ai_triz.config.settings import get_settings

    settings = get_settings()
    openai_api_key = settings.openai.api_key
"""

import os
from typing import Optional, List
from functools import lru_cache
import requests
from pydantic_settings import BaseSettings

class LLMProviderSettings(BaseSettings):
    """
    Base settings for LLM providers.

    Attributes:
        temperature (float): The sampling temperature for the LLM. Defaults to 0.8.
        max_tokens (Optional[int]): The maximum number of tokens to generate. Defaults to None.
        max_retries (int): The maximum number of retries for API calls. Defaults to 3.
    """
    temperature: float = 0.8
    max_tokens: Optional[int] = 1024
    max_retries: int = 3

class OpenAISettings(LLMProviderSettings):
    """
    Settings specific to OpenAI.

    Attributes:
        api_key (str): The API key for OpenAI. Retrieved from environment variable.
        default_model (str): The default model to use for OpenAI. Defaults to "gpt-4o-mini-2024-07-18".
    """
    default_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Fetch the list of available models from the Groq API."""
        api_key = os.getenv("OPENAI_API_KEY")
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        except requests.RequestException as e:
            print(f"Error fetching available models: {e}")
            return []

class OllamaSettings(LLMProviderSettings):
    """
    Settings specific to Ollama.

    Attributes:
        api_key (str): A placeholder API key. Required but not used.
        default_model (str): The default model to use for Ollama. Defaults to "llama3:instruct".
        base_url (str): The base URL for Ollama API. Defaults to "http://localhost:11434/v1".
    """
    default_model: str = "llama3.1:8b"
    embedding_model: str = "mxbai-embed-large:335m"
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Fetch available models from local Ollama."""
        try:
            response = requests.get(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            models = response.json()
            return [model['name'] for model in models['models']]
        except requests.RequestException as e:
            print(f"Error fetching Ollama models: {e}")
            return []

class GroqSettings(LLMProviderSettings):
    """
    Settings specific to Groq.

    Attributes:
        api_key (str): The API key for Groq. Retrieved from environment variable.
        default_model (str): The default model to use for Groq. Defaults to "llama3-8b-8192".
        base_url (str): The base URL for Groq API. Defaults to "https://api.groq.com/openai/v1".
    """
    default_model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Fetch the list of available models from the Groq API."""
        api_key = os.getenv("GROQ_API_KEY")
        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers=
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        except requests.RequestException as e:
            print(f"Error fetching available models: {e}")
            return []

class AnthropicSettings(LLMProviderSettings):
    """
    Settings specific to Anthropic.
    """
    default_model: str = "claude-3-5-sonnet-latest"

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Fetch available models from local Anthropic."""
        return ["claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022"]

class HuggingFaceSettings:
    """Settings for Hugging Face models"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def get_available_models(self) -> List[str]:
        """Get list of available Hugging Face models"""
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]

class Settings(BaseSettings):
    """
    Main settings class for the application.

    Attributes:
        openai (OpenAISettings): Settings for OpenAI.
        ollama (OllamaSettings): Settings for Ollama.
        groq (GroqSettings): Settings for Groq.
        anthropic (AnthropicSettings): Settings for Anthropic.
        huggingface (HuggingFaceSettings): Settings for Hugging Face models.
    """
    openai: OpenAISettings = OpenAISettings()
    ollama: OllamaSettings = OllamaSettings()
    groq: GroqSettings = GroqSettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    huggingface: HuggingFaceSettings = HuggingFaceSettings()

@lru_cache
def get_settings():
    """
    Retrieve the application settings.

    This function uses lru_cache decorator to memoize the settings,
    ensuring that the settings are only loaded once and then cached
    for subsequent calls, improving performance.

    Returns:
        Settings: An instance of the Settings class containing all configuration parameters for the application.
    """
    return Settings()

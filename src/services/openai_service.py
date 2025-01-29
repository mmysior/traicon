"""
OpenAI service.
"""
import os
from typing import Dict, List, Type
import instructor

from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from src.config.settings import get_settings

# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 1

# -----------------------------------------------------------------------------
# Basic functions
# -----------------------------------------------------------------------------

class BaseService:
    """
    Base service class.
    """
    def __init__(self, provider: str = 'openai'):
        """
        Initialize the OpenAI service.
        """
        load_dotenv()
        self.provider = provider
        self.client = self._get_client()
        self.available_models = self._get_available_models()
        self.default_model = self._get_default_model()

    def _get_client(self):
        """
        Get the client for OpenAI, Groq or Ollama.
        """
        if self.provider == 'openai':
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=get_settings().openai.base_url
            )
        elif self.provider == 'groq':
            return OpenAI(
                base_url=get_settings().groq.base_url,
                api_key=os.getenv("GROQ_API_KEY")
            )
        elif self.provider == 'ollama':
            return OpenAI(
                base_url=get_settings().ollama.base_url,
                api_key=os.getenv("OLLAMA_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_available_models(self):
        """
        Get the list of available models based on the provider.
        """
        if self.provider == 'openai':
            return get_settings().openai.get_available_models()
        elif self.provider == 'groq':
            return get_settings().groq.get_available_models()
        elif self.provider == 'ollama':
            return get_settings().ollama.get_available_models()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_default_model(self):
        """
        Get the default model based on the provider.
        """
        if self.provider == 'openai':
            if get_settings().openai.default_model in self.available_models:
                return get_settings().openai.default_model
        elif self.provider == 'groq':
            if get_settings().groq.default_model in self.available_models:
                return get_settings().groq.default_model
        elif self.provider == 'ollama':
            if get_settings().ollama.default_model in self.available_models:
                return get_settings().ollama.default_model
        else:
            raise ValueError(f"Incorrect default model for provider: {self.provider}")

class OpenAIService(BaseService):
    """
    OpenAI service.
    """
    @observe(name="create_chat_completion", as_type="generation", capture_input=False, capture_output=False)
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict:
        """
        Basic answer to a question.
        """
        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)
        model = kwargs.pop('model', self.default_model)

        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        # Update Langfuse context with input
        langfuse_context.update_current_observation(
            input=messages,
            model=model,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                **kwargs
            }
        )

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Update Langfuse context with output
        langfuse_context.update_current_observation(
            output=[choice.message.content for choice in response.choices],
            usage={
                'input': response.usage.prompt_tokens,
                'output': response.usage.completion_tokens,
            }
        )

        return response

    def get_answer(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get only the answer from the chat completion.
        """
        response = self.create_chat_completion(messages, **kwargs)
        return response.choices[0].message.content

    def get_answer_with_metadata(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Get the answer and metadata from the chat completion.
        """
        response = self.create_chat_completion(messages, **kwargs)
        return {
            'answer': response.choices[0].message.content,
            'metadata': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
        }

    @observe(name="create_structured_completion", as_type="generation", capture_input=False, capture_output=False)
    def create_structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Create a chat completion with structured output using Instructor.

        Args:
            messages: List of message dictionaries
            response_model: Pydantic model class defining the response structure
            **kwargs: Additional arguments for the completion

        Returns:
            BaseModel: The structured response
        """
        model = kwargs.pop('model', self.default_model)
        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        kwargs.pop('n', None)
        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)

        # Update Langfuse context with input
        langfuse_context.update_current_observation(
            input=messages,
            model=model,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                **kwargs
            }
        )

        patched_client = instructor.from_openai(self.client)
        response, completion = patched_client.chat.completions.create_with_completion(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=1,
            **kwargs
        )

        # Update Langfuse context with output
        langfuse_context.update_current_observation(
            output=[response.model_dump()] if isinstance(response, BaseModel) else [r.model_dump() for r in response],
            usage={
                'input': completion.usage.prompt_tokens,
                'output': completion.usage.completion_tokens,
            }
        )

        return response

class AsyncOpenAIService(BaseService):
    """
    Async OpenAI service.
    """
    def _get_client(self):
        """
        Get the async client for OpenAI, Groq or Ollama.
        """
        if self.provider == 'openai':
            return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == 'groq':
            return AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
        elif self.provider == 'ollama':
            return AsyncOpenAI(base_url=get_settings().ollama.base_url)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @observe(name="create_chat_completion", as_type="generation", capture_input=False, capture_output=False)
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict:
        """
        Async version of chat completion.
        """
        model = kwargs.pop('model', self.default_model)
        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)

        # Update Langfuse context with input
        langfuse_context.update_current_observation(
            input=messages,
            model=model,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                **kwargs
            }
        )

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Update Langfuse context with output
        langfuse_context.update_current_observation(
            output=[choice.message.content for choice in response.choices],
            usage={
                'input': response.usage.prompt_tokens,
                'output': response.usage.completion_tokens,
            }
        )

        return response

    async def get_answer(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get only the answer from the chat completion.
        """
        response = await self.create_chat_completion(messages, **kwargs)
        return response.choices[0].message.content

    async def get_answer_with_metadata(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict:
        """
        Get the answer and metadata from the chat completion.
        """
        response = await self.create_chat_completion(messages, **kwargs)
        return {
            'answer': response.choices[0].message.content,
            'metadata': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
        }

    @observe(name="create_structured_completion", as_type="generation", capture_input=False, capture_output=False)
    async def create_structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Create a chat completion with structured output using Instructor.

        Args:
            messages: List of message dictionaries
            response_model: Pydantic model class defining the response structure
            **kwargs: Additional arguments for the completion

        Returns:
            BaseModel: The structured response
        """
        model = kwargs.pop('model', self.default_model)
        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        # Remove n from kwargs if it exists and set our parameters
        kwargs.pop('n', None)
        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)

        # Update Langfuse context with input
        langfuse_context.update_current_observation(
            input=messages,
            model=model,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                **kwargs
            }
        )

        patched_client = instructor.from_openai(self.client)
        response, completion = await patched_client.chat.completions.create_with_completion(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=1,
            **kwargs
        )

        # Update Langfuse context with output
        langfuse_context.update_current_observation(
            output=[response.model_dump()] if isinstance(response, BaseModel) else [r.model_dump() for r in response],
            usage={
                'input': completion.usage.prompt_tokens,
                'output': completion.usage.completion_tokens,
            }
        )

        return response

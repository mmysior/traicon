"""
Anthropic service.
"""
import os
from typing import Dict, List, Type

import instructor
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
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
    def __init__(self):
        """
        Initialize the Anthropic service.
        """
        load_dotenv()
        self.client = self._get_client()
        self.available_models = self._get_available_models()
        self.default_model = self._get_default_model()

    def _get_client(self):
        """
        Get the appropriate client (sync/async).
        Should be implemented by child classes.
        """
        raise NotImplementedError

    def _get_available_models(self):
        """
        Get the list of available models.
        """
        return get_settings().anthropic.get_available_models()

    def _get_default_model(self):
        """
        Get the default model.
        """
        default_model = get_settings().anthropic.default_model
        if default_model not in self.available_models:
            raise ValueError(f"Incorrect default model: {default_model}")
        return default_model

class AnthropicService(BaseService):
    """
    Synchronous Anthropic service.
    """
    def _get_client(self):
        """
        Get the synchronous Anthropic client.
        """
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @observe(name="create_chat_completion", as_type="generation", capture_input=False, capture_output=False)
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict:
        """
        Basic answer to a question.
        """
        model = kwargs.pop('model', self.default_model)
        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)

        # Update Langfuse context
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

        # Get response from Anthropic
        response = self.client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Update Langfuse context
        langfuse_context.update_current_observation(
            output=response.content[0].text,
            usage={
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens,
            }
        )

        return response

    def get_answer(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get only the answer from the chat completion.
        """
        response = self.create_chat_completion(messages, **kwargs)
        return response.content[0].text

    def get_answer_with_metadata(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Get the answer and metadata from the chat completion.
        """
        response = self.create_chat_completion(messages, **kwargs)
        return {
            'answer': response.content[0].text,
            'metadata': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
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
            Tuple of (structured_response, raw_completion)
        """
        model = kwargs.pop('model', self.default_model)
        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)

        # Update Langfuse context
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

        patched_client = instructor.from_anthropic(self.client)
        response, completion = patched_client.messages.create_with_completion(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Update Langfuse context
        langfuse_context.update_current_observation(
            output=[response.model_dump()] if isinstance(response, BaseModel) else [r.model_dump() for r in response],
            usage={
                'input': completion.usage.input_tokens,
                'output': completion.usage.output_tokens,
            }
        )

        return response

class AsyncAnthropicService(BaseService):
    """
    Asynchronous Anthropic service.
    """
    def _get_client(self):
        """
        Get the asynchronous Anthropic client.
        """
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

        # Update Langfuse context
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

        response = await self.client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Update Langfuse context
        langfuse_context.update_current_observation(
            output=response.content[0].text,
            usage={
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens,
            }
        )

        return response

    async def get_answer(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get only the answer from the chat completion.
        """
        response = await self.create_chat_completion(messages, **kwargs)
        return response.content[0].text

    async def get_answer_with_metadata(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Get the answer and metadata from the chat completion.
        """
        response = await self.create_chat_completion(messages, **kwargs)
        return {
            'answer': response.content[0].text,
            'metadata': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
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
            Tuple of (structured_response, raw_completion)
        """
        model = kwargs.pop('model', self.default_model)
        if model not in self.available_models:
            raise ValueError(f"Unsupported model: {model}")

        temperature = kwargs.pop('temperature', DEFAULT_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', DEFAULT_MAX_TOKENS)
        top_p = kwargs.pop('top_p', DEFAULT_TOP_P)

        # Update Langfuse context
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

        patched_client = instructor.from_anthropic(self.client)
        response, completion = await patched_client.messages.create_with_completion(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Update Langfuse context
        langfuse_context.update_current_observation(
            output=[response.model_dump()] if isinstance(response, BaseModel) else [r.model_dump() for r in response],
            usage={
                'input': completion.usage.input_tokens,
                'output': completion.usage.output_tokens,
            }
        )

        return response

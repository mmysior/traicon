"""
This module provides embedding services with support for multiple providers like Ollama and OpenAI.
"""
import os
from typing import List, Dict
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings

class BaseService:
    """
    Base service class.
    """
    def __init__(self, provider: str = 'openai'):
        """
        Initialize the service with specified provider.
        
        Args:
            provider: Service provider ('openai', 'ollama', or 'huggingface')
        """
        load_dotenv()
        self.provider = provider
        self.client = self._get_client()
        self.available_models = self._get_available_models()
        self.model = self._get_embedding_model()

    def _get_client(self):
        """
        Get the client for OpenAI, Groq, Ollama or Hugging Face.
        """
        if self.provider == 'openai':
            return OpenAI()
        elif self.provider == 'ollama':
            return OpenAI(
                base_url=get_settings().ollama.base_url,
                api_key=os.getenv("OLLAMA_API_KEY")
            )
        elif self.provider == 'huggingface':
            model_name = get_settings().huggingface.embedding_model
            return SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_available_models(self):
        """
        Get the list of available models based on the provider.
        """
        if self.provider == 'openai':
            return get_settings().openai.get_available_models()
        elif self.provider == 'ollama':
            return get_settings().ollama.get_available_models()
        elif self.provider == 'huggingface':
            return get_settings().huggingface.get_available_models()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_embedding_model(self):
        """
        Get the embedding model based on the provider.
        """
        if self.provider == 'openai':
            embedding_model = get_settings().openai.embedding_model
        elif self.provider == 'huggingface':
            embedding_model = get_settings().huggingface.embedding_model
        else:
            embedding_model = get_settings().ollama.embedding_model

        if embedding_model not in self.available_models:
            raise ValueError(f"Invalid embedding model: {embedding_model}")
        return embedding_model

class EmbeddingService(BaseService):
    """
    Service for creating and comparing embeddings.
    Provides methods to create embeddings and find similar vectors.
    """
    def create_embedding(self, text: str, model: str | None = None) -> List[float]:
        """
        Create embedding using the configured provider.

        Args:
            text: Input text to create embedding for
            model: Optional model override. Uses default if not specified

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding creation fails

        Returns:
            List of floating point values representing the embedding
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text input must be a non-empty string")

        model = model or self.model
        try:
            if self.provider == 'huggingface':
                # Hugging Face sentence-transformers return numpy array
                embedding = self.client.encode(text)
                return embedding.tolist()

            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to create embedding: {str(e)}") from e

    def find_n_closest(
        self,
        query_vector: List[float],
        embeddings: List[List[float]],
        n: int = 3
    ) -> List[Dict[str, float]]:
        """
        Find the n closest vectors to the query vector using cosine distance.

        Args:
            query_vector: Vector to compare against
            embeddings: List of vectors to search through
            n: Number of closest matches to return

        Returns:
            List of dicts containing index and distance of closest matches

        Raises:
            ValueError: If query_vector and embeddings have different dimensions
        """
        query_len = len(query_vector)
        for i, vec in enumerate(embeddings):
            if len(vec) != query_len:
                raise ValueError(
                    f"Embedding at index {i} has length {len(vec)}, "
                    f"but query vector has length {query_len}"
                )

        distances = [
            {"index": i, "distance": distance.cosine(query_vector, vec)}
            for i, vec in enumerate(embeddings)
        ]
        return sorted(distances, key=lambda x: x["distance"])[:n]

class AsyncEmbeddingService(BaseService):
    """
    Async service for creating and comparing embeddings.
    Provides asynchronous methods for embedding operations.
    """
    def _get_client(self):
        """
        Get the async client for OpenAI, Groq or Ollama.
        """
        if self.provider == 'openai':
            return AsyncOpenAI()
        elif self.provider == 'ollama':
            return AsyncOpenAI(base_url=get_settings().ollama.base_url, api_key="ollama")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    async def create_embedding(self, text: str, model: str | None = None) -> List[float]:
        """
        Create embedding asynchronously using the configured provider.

        Args:
            text: Input text to create embedding for
            model: Optional model override. Uses default if not specified

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding creation fails

        Returns:
            List of floating point values representing the embedding
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text input must be a non-empty string")

        model = model or self.model
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to create embedding: {str(e)}") from e

    async def find_n_closest(
        self,
        query_vector: List[float],
        embeddings: List[List[float]],
        n: int = 3
    ) -> List[Dict[str, float]]:
        """
        Find the n closest vectors to the query vector using cosine distance.
        """
        distances = [
            {"index": i, "distance": distance.cosine(query_vector, vec)}
            for i, vec in enumerate(embeddings)
        ]
        return sorted(distances, key=lambda x: x["distance"])[:n]

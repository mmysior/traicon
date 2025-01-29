"""
This script serves as a builder of the database needed to process TRIZ Contradictions
"""
import os
import json
import asyncio
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from src.services.embedding_service import AsyncEmbeddingService

parameters_txt_path = os.path.join("src", "resources", "parameters.txt")
parameters_json_path = os.path.join("src", "resources", "parameters.json")

# Load the environment variables
load_dotenv()

# Define embedding service
embedding_service = AsyncEmbeddingService(provider='openai')

# Maximum concurrent requests
MAX_CONCURRENT_REQUESTS = 5

async def embed_parameters(parameters: List[str]) -> List[dict]:
    """
    Embed the TRIZ standard parameters concurrently.
    
    Args:
        parameters: List of parameter strings to embed
        
    Returns:
        List of dictionaries containing parameter info and embeddings
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_parameter(ind: int, parameter: str) -> dict:
        async with semaphore:
            return {
                "uuid": str(uuid4()),
                "index": ind + 1,
                "parameter": parameter,
                "embedding": {
                    "model": embedding_service.embedding_model,
                    "vector": await embedding_service.create_embedding(parameter),
                }
            }

    tasks = [
        process_parameter(ind, parameter)
        for ind, parameter in enumerate(parameters)
    ]

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=f"Embedding parameters using {embedding_service.embedding_model}"
    )

    return results

async def main():
    """
    Main function.
    """

    # Check if embeddings file exists
    if os.path.exists(parameters_json_path):
        print("Loading existing embeddings from JSON file...")
        try:
            with open(parameters_json_path, "r", encoding='utf-8') as file:
                parameters = json.load(file)
            print(f"Successfully loaded {len(parameters)} embeddings")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load existing embeddings: {e}")
            return
    else:
        print("Embeddings file not found. Generating new embeddings...")
        # Load the TRIZ standard parameters
        try:
            with open(parameters_txt_path, "r", encoding='utf-8') as file:
                parameters_txt = [line.strip() for line in file.readlines()]
            print(f"Successfully loaded {len(parameters_txt)} parameters from text file")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load parameters from text file: {e}")
            return

        # Generate and save embeddings
        try:
            parameters = await embed_parameters(parameters_txt)
            with open(parameters_json_path, "w", encoding='utf-8') as file:
                json.dump(parameters, file, indent=4)
            print(f"Successfully generated and saved {len(parameters)} embeddings to {parameters_json_path}")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to generate and save embeddings: {e}")
            return

if __name__ == '__main__':
    asyncio.run(main())

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

principles_txt_path = os.path.join("src", "resources", "principles.txt")
principles_json_path = os.path.join("src", "resources", "principles.json")

# Load the environment variables
load_dotenv()

# Define embedding service
embedding_service = AsyncEmbeddingService(provider='openai')
# Maximum concurrent requests
MAX_CONCURRENT_REQUESTS = 5

async def embed_principles(principles: List[str]) -> List[dict]:
    """
    Embed the TRIZ principles concurrently.
    
    Args:
        principles: List of principle strings to embed
        
    Returns:
        List of dictionaries containing principle info and embeddings
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_principle(ind: int, principle: str) -> dict:
        async with semaphore:
            # Split the principle into name and description
            name, description = principle.split(':', 1)
            name = name.strip()
            description = description.strip()
            
            return {
                "uuid": str(uuid4()),
                "index": ind + 1,
                "name": name,
                "description": description,
                "embedding": {
                    "model": embedding_service.embedding_model,
                    "vector": await embedding_service.create_embedding(principle),
                }
            }

    tasks = [
        process_principle(ind, principle)
        for ind, principle in enumerate(principles)
    ]

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=f"Embedding principles using {embedding_service.embedding_model}"
    )

    return results

async def main():
    """
    Main function.
    """

    # Check if embeddings file exists
    if os.path.exists(principles_json_path):
        print("Loading existing embeddings from JSON file...")
        try:
            with open(principles_json_path, "r", encoding='utf-8') as file:
                principles = json.load(file)
            print(f"Successfully loaded {len(principles)} embeddings")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load existing embeddings: {e}")
            return
    else:
        print("Embeddings file not found. Generating new embeddings...")
        # Load the TRIZ principles
        try:
            with open(principles_txt_path, "r", encoding='utf-8') as file:
                principles_txt = [line.strip() for line in file.readlines() if line.strip()]
            print(f"Successfully loaded {len(principles_txt)} principles from text file")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load principles from text file: {e}")
            return

        # Generate and save embeddings
        try:
            principles = await embed_principles(principles_txt)
            with open(principles_json_path, "w", encoding='utf-8') as file:
                json.dump(principles, file, indent=4)
            print(f"Successfully generated and saved {len(principles)} embeddings to {principles_json_path}")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to generate and save embeddings: {e}")
            return

if __name__ == '__main__':
    asyncio.run(main())

"""
This module provides functionality for processing technical contradictions in inventive problems.

It includes:
- Classification of problem descriptions to identify technical contradictions
- Extraction of contradiction components (action, positive/negative effects)
- Mapping to standard engineering parameters
- Generation of relevant inventive principles

The module uses LLM services to analyze problem descriptions and extract key components.
Parameters and principles are mapped using the TRIZ contradiction matrix.

"""
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel

from src.services.openai_service import OpenAIService
from src.services.embedding_service import EmbeddingService
from src.utils.matrix import load_params_data, get_inventive_principles
from app.prompts.tc_prompts import TCExtractionPrompt, ClassificationPrompt

# -------------------------------------------------------------------------------------------------
# Instantiate services
# -------------------------------------------------------------------------------------------------

openai_service = OpenAIService()
embedding_service = EmbeddingService()

# -------------------------------------------------------------------------------------------------
# Define classes
# -------------------------------------------------------------------------------------------------

class TechnicalContradiction(BaseModel):
    """Pydantic model for the technical contradiction analysis output"""
    uuid: str
    problem_desc: str
    action: str
    positive_effect: str
    negative_effect: str
    parameters_to_improve: List[int]
    parameters_to_preserve: List[int]
    principles: List[int]

class TCComponents(BaseModel):
    """Simple Pydantic Model for Technical Contradiction components"""
    action: str
    positive_effect: str
    negative_effect: str

# -------------------------------------------------------------------------------------------------
# Define functions
# -------------------------------------------------------------------------------------------------

def classify_problem(problem_desc: str, model: str = "gpt-4o-mini", **kwargs: Any) -> bool:
    """
    Classify whether input text describes a technical contradiction.

    Args:
        problem_desc: Text to classify
        platform: LLM platform to use (default: 'ollama')
        **kwargs: Additional arguments passed to LLM

    Returns:
        bool: True if text describes a technical contradiction, False otherwise
    """
    messages = ClassificationPrompt.compile_messages(
        query=problem_desc
    )

    response = openai_service.get_answer(
        messages=messages,
        model=model,
        **kwargs
    )
    return response.lower() == "true"

# -------------------------------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------------------------------

def formulate_tc(
    problem_desc: str,
    n: int = 1,
    model: str = "gpt-4o-mini",
    **kwargs: Any
) -> TechnicalContradiction:
    """
    Process a technical contradiction from problem description to final analysis.
    
    Args:
        problem_desc: Description of the technical problem
        n: Number of parameters to return for each effect
        model: LLM model to use
        **kwargs: Additional arguments for the LLM
        
    Returns:
        TechnicalContradictionOutput: Complete analysis of the technical contradiction
    """
    # Extract contradiction components
    contradiction_parts = openai_service.create_structured_completion(
        messages=TCExtractionPrompt().compile_messages(query=problem_desc),
        response_model=TCComponents,
        model=model,
        **kwargs
    )

    # Get embeddings and search for parameters
    positive_effect_embedding = embedding_service.create_embedding(contradiction_parts.positive_effect)
    negative_effect_embedding = embedding_service.create_embedding(contradiction_parts.negative_effect)

    parameters = load_params_data()

    def find_closest_parameters(embedding: List[float], n: int) -> List[Dict[str, Any]]:
        distances = embedding_service.find_n_closest(
            query_vector=embedding,
            embeddings=[param["embedding"]["vector"] for param in parameters],
            n=n,
        )
        return [parameters[dist["index"]]["index"] for dist in distances]

    parameters_to_improve = find_closest_parameters(positive_effect_embedding, n)
    parameters_to_preserve = find_closest_parameters(negative_effect_embedding, n)

    # Get principles
    principles = get_inventive_principles(
        improving_parameter=parameters_to_improve,
        preserving_parameter=parameters_to_preserve
    )

    return TechnicalContradiction(
        uuid=str(uuid4()),
        problem_desc=problem_desc,
        action=contradiction_parts.action,
        positive_effect=contradiction_parts.positive_effect,
        negative_effect=contradiction_parts.negative_effect,
        parameters_to_improve=parameters_to_improve,
        parameters_to_preserve=parameters_to_preserve,
        principles=principles
    )

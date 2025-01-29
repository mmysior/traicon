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
from typing import Any, List
from uuid import uuid4

from langfuse.decorators import observe, langfuse_context
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

class TCModel(BaseModel):
    """Simple Pydantic Model for Technical Contradiction components"""
    action: str
    positive_effect: str
    negative_effect: str

# -------------------------------------------------------------------------------------------------
# Define functions
# -------------------------------------------------------------------------------------------------

@observe(name="classify_problem")
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

@observe(name="formulate_tc", as_type="generation", capture_input=False)
def formulate_tc(problem_desc: str, model: str = "gpt-4o-mini", **kwargs: Any) -> TCModel:
    """
    Formulate a technical contradiction from a problem description.
    """
    kwargs_clone = kwargs.copy()
    messages = TCExtractionPrompt().compile_messages(query=problem_desc)

    langfuse_context.update_current_observation(
        input=messages,
        model=model,
        metadata=kwargs_clone
    )
    return openai_service.create_structured_completion(
        messages=messages,
        response_model=TCModel,
        model=model,
        **kwargs
    )

@observe(name="assign_parameters")
def assign_parameters(description: str, n: int = 1) -> List[int]:
    """
    Assign parameters to a description.
    """
    embedding = embedding_service.create_embedding(description)
    parameters = load_params_data()

    distances = embedding_service.find_n_closest(
        query_vector=embedding,
        embeddings=[param["embedding"]["vector"] for param in parameters],
        n=n,
    )

    return [parameters[dist["index"]]["index"] for dist in distances]

@observe(name="get_principles")
def get_principles(improving_parameter: List[int], preserving_parameter: List[int]) -> List[int]:
    """
    Get principles from a list of parameters.
    """
    return get_inventive_principles(improving_parameter, preserving_parameter)

# -------------------------------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------------------------------

@observe(name="process_tc")
def process_tc(
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
        TechnicalContradiction: Complete analysis of the technical contradiction
    """
    # Extract contradiction components
    tc_model = formulate_tc(problem_desc, model=model, **kwargs)

    # Get parameters for positive and negative effects
    parameters_to_improve = assign_parameters(tc_model.positive_effect, n=n)
    parameters_to_preserve = assign_parameters(tc_model.negative_effect, n=n)

    # Get principles
    principles = get_principles(
        improving_parameter=parameters_to_improve,
        preserving_parameter=parameters_to_preserve
    )

    return TechnicalContradiction(
        uuid=str(uuid4()),
        problem_desc=problem_desc,
        action=tc_model.action,
        positive_effect=tc_model.positive_effect,
        negative_effect=tc_model.negative_effect,
        parameters_to_improve=parameters_to_improve,
        parameters_to_preserve=parameters_to_preserve,
        principles=principles
    )

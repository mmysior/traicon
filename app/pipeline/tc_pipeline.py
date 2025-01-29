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
from typing import Any, List, Dict
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

@observe(name="formulate_tc", capture_input=False)
def formulate_tc(problem_desc: str, model: str = "gpt-4o-mini", **kwargs: Any) -> TCModel:
    """
    Formulate a technical contradiction from a problem description.
    """
    kwargs_clone = kwargs.copy()
    messages = TCExtractionPrompt().compile_messages(query=problem_desc)

    langfuse_context.update_current_observation(
        input=problem_desc,
        model=model,
        metadata=kwargs_clone
    )

    return openai_service.create_structured_completion(
        messages=messages,
        response_model=TCModel,
        model=model,
        **kwargs
    )

@observe(name="assign_parameters", capture_input=False)
def assign_parameters(positive_effect: str, negative_effect: str, n: int = 1) -> Dict[str, List[int]]:
    """
    Assign parameters to a description.
    """
    def find_parameters(text: str, n: int = 1) -> List[int]:
        """
        Assign parameters to a description.
        """
        embedding = embedding_service.create_embedding(text)
        parameters = load_params_data()

        distances = embedding_service.find_n_closest(
            query_vector=embedding,
            embeddings=[param["embedding"]["vector"] for param in parameters],
            n=n,
        )

        return [parameters[dist["index"]]["index"] for dist in distances]

    langfuse_context.update_current_observation(
        input={
            "positive_effect": positive_effect,
            "negative_effect": negative_effect,
        },
        metadata={"n": n},
    )

    improving_parameters = find_parameters(positive_effect, n=n)
    preserving_parameters = find_parameters(negative_effect, n=n)

    return {
        "improving_parameters": improving_parameters,
        "preserving_parameters": preserving_parameters,
    }

@observe(name="get_principles", capture_input=False, capture_output=False)
def get_principles(improving_parameters: List[int], preserving_parameters: List[int]) -> List[int]:
    """
    Get principles from a list of parameters.
    """
    inventive_principles = get_inventive_principles(improving_parameters, preserving_parameters)

    langfuse_context.update_current_observation(
        input={"improving_parameters": improving_parameters, "preserving_parameters": preserving_parameters},
        output=inventive_principles,
    )

    return inventive_principles

# -------------------------------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------------------------------

@observe(name="process_tc", capture_input=False)
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
    kwargs_clone = kwargs.copy()

    langfuse_context.update_current_observation(
        input=problem_desc,
        model=model,
        metadata=kwargs_clone
    )

    tc_model = formulate_tc(problem_desc, model=model, **kwargs_clone)

    # Get parameters for positive and negative effects
    standard_parameters = assign_parameters(tc_model.positive_effect, tc_model.negative_effect, n=n)

    # Get principles
    principles = get_principles(
        improving_parameters=standard_parameters["improving_parameters"],
        preserving_parameters=standard_parameters["preserving_parameters"]
    )

    return TechnicalContradiction(
        uuid=str(uuid4()),
        problem_desc=problem_desc,
        action=tc_model.action,
        positive_effect=tc_model.positive_effect,
        negative_effect=tc_model.negative_effect,
        parameters_to_improve=standard_parameters["improving_parameters"],
        parameters_to_preserve=standard_parameters["preserving_parameters"],
        principles=principles
    )

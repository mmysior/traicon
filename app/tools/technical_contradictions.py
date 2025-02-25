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
from typing import Any, List, Dict, Union
from uuid import uuid4

from langfuse.decorators import observe, langfuse_context
from pydantic import BaseModel

from app.prompts.tc_prompts import TCExtractionPrompt, ClassificationPrompt, TCSolutionPrompt
from src.services.openai_service import OpenAIService
from src.services.anthropic_service import AnthropicService
from src.services.embedding_service import EmbeddingService
from src.utils.matrix import (
    load_parameters,
    load_principles,
    make_parameters,
    make_principles,
    get_inventive_principles
)

# -------------------------------------------------------------------------------------------------
# Define classes
# -------------------------------------------------------------------------------------------------

class TCModel(BaseModel):
    """Simple Pydantic Model for Technical Contradiction components"""
    action: str
    positive_effect: str
    negative_effect: str

class Solution(BaseModel):
    """
    Pydantic model for a single solution
    """
    uuid: str
    principle_id: int
    principle_name: str
    solution: str

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
    solutions: List[Solution]

# -------------------------------------------------------------------------------------------------
# Define functions
# -------------------------------------------------------------------------------------------------

@observe(name="ClassifyProblem", capture_input=False)
def classify_problem(
    problem_desc: str,
    llm_service: Union[OpenAIService, AnthropicService],
    model: str = "gpt-4o-mini",
    **kwargs: Any
    ) -> bool:
    """
    Classify whether input text describes a technical contradiction.

    Args:
        problem_desc: Text to classify
        llm_service: LLM service to use
        model: LLM model to use (default: 'gpt-4o-mini')
        **kwargs: Additional arguments passed to LLM

    Returns:
        bool: True if text describes a technical contradiction, False otherwise
    """
    messages = ClassificationPrompt.compile_messages(
        query=problem_desc
    )

    response = llm_service.get_answer(
        messages=messages,
        model=model,
        **kwargs
    )
    return response.lower() == "true"

@observe(name="ExtractTC", capture_input=False)
def extract_tc(
    problem_desc: str,
    llm_service: Union[OpenAIService, AnthropicService],
    model: str = "gpt-4o-mini",
    **kwargs: Any
    ) -> TCModel:
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

    return llm_service.create_structured_completion(
        messages=messages,
        response_model=TCModel,
        model=model,
        **kwargs
    )

@observe(name="AssignParameters", capture_input=False)
def assign_parameters(
    positive_effect: str,
    negative_effect: str,
    embedding_service: EmbeddingService,
    n: int = 1
    ) -> Dict[str, List[int]]:
    """
    Assign parameters to a description.
    """
    # Load or generate embeddings with correct model
    try:
        parameters = load_parameters()
        if parameters['metadata']['model'] != embedding_service.model:
            parameters = make_parameters(embedding_service)
    except FileNotFoundError:
        parameters = make_parameters(embedding_service)

    def find_parameters(text: str, n: int = 1) -> List[int]:
        """
        Assign parameters to a description.
        """
        embedding = embedding_service.create_embedding(text)

        distances = embedding_service.find_n_closest(
            query_vector=embedding,
            embeddings=[param["embedding"] for param in parameters['data']],
            n=n,
        )

        return [parameters['data'][dist["index"]]["index"] for dist in distances]

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

@observe(name="GetIPs", capture_input=False, capture_output=False)
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

@observe(name="GenerateSolutions", capture_input=False)
def generate_solutions(
    problem_desc: str,
    selected_principles: List[int],
    llm_service: Union[OpenAIService, AnthropicService],
    embedding_service: EmbeddingService,
    model: str = "gpt-4o-mini",
    **kwargs: Any
) -> List[Solution]:
    """
    Generate solutions for a technical contradiction.
    
    Args:
        problem_desc: Description of the technical problem
        principles: List of principle indices to generate solutions for
        model: LLM model to use
        **kwargs: Additional arguments for the LLM
        
    Returns:
        List[Solution]: List of generated solutions
    """
    kwargs_clone = kwargs.copy()

    langfuse_context.update_current_observation(
        input={
            "problem_desc": problem_desc,
            "principles": selected_principles
        },
        model=model,
        metadata=kwargs_clone
    )
    # Try to load principles, create if not found
    try:
        principles = load_principles()
    except FileNotFoundError:
        principles = make_principles(embedding_service)

    solutions = []
    for ip_index in selected_principles:
        # Retrieve inventive principle info
        ip_description = principles['data'][ip_index - 1]['description']
        ip_name = principles['data'][ip_index - 1]['name']

        # Build context
        context = f"Inventive Principle Name: {ip_name}\nDescription: {ip_description}"
        messages = TCSolutionPrompt.compile_messages(context=context, query=problem_desc)

        # Generate solution
        response = llm_service.get_answer(
            messages=messages,
            model=model,
            **kwargs
        )

        # Create Solution object
        solution = Solution(
            uuid=str(uuid4()),
            principle_id=ip_index,
            principle_name=ip_name,
            solution=response
        )
        solutions.append(solution)

    return solutions

# -------------------------------------------------------------------------------------------------
# Pipelines
# -------------------------------------------------------------------------------------------------

@observe(name="SolveTC", capture_input=False)
def solve_tc(
    problem_desc: str,
    llm_service: Union[OpenAIService, AnthropicService],
    embedding_service: EmbeddingService,
    n: int = 1,
    model: str = "gpt-4o-mini",
    **kwargs: Any
) -> TechnicalContradiction:
    """
    Process a technical contradiction from a problem description to final analysis.
    
    Args:
        problem_desc: Description of the technical problem
        n: Number of parameters to return for each effect
        model: LLM model to use
        **kwargs: Additional arguments for the LLM
        
    Returns:
        TechnicalContradiction: Technical contradiction components
    """
    kwargs_clone = kwargs.copy()

    langfuse_context.update_current_observation(
        input=problem_desc,
        model=model,
        metadata=kwargs_clone
    )

    # Extract contradiction components - make this async
    tc_model = extract_tc(problem_desc, llm_service, model=model, **kwargs_clone)

    # These operations are synchronous and can remain as is
    standard_parameters = assign_parameters(tc_model.positive_effect, tc_model.negative_effect, embedding_service, n=n)
    principles = get_principles(
        improving_parameters=standard_parameters["improving_parameters"],
        preserving_parameters=standard_parameters["preserving_parameters"]
    )

    # Generate solutions asynchronously
    solutions = generate_solutions(
        problem_desc=problem_desc,
        selected_principles=principles,
        llm_service=llm_service,
        embedding_service=embedding_service,
        model=model,
        **kwargs_clone
    )

    return TechnicalContradiction(
        uuid=str(uuid4()),
        problem_desc=problem_desc,
        action=tc_model.action,
        positive_effect=tc_model.positive_effect,
        negative_effect=tc_model.negative_effect,
        parameters_to_improve=standard_parameters["improving_parameters"],
        parameters_to_preserve=standard_parameters["preserving_parameters"],
        principles=principles,
        solutions=solutions
    )

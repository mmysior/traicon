"""
Generator module for creating solutions to technical contradictions using TRIZ principles
and LLM-based generation.
"""
from typing import Any, List
from uuid import uuid4
import asyncio

from langfuse.decorators import observe, langfuse_context
from pydantic import BaseModel

from src.services.openai_service import AsyncOpenAIService
from src.utils.matrix import get_principle_description, get_principle_name
from app.prompts.generate_solution import GenerateSolutionPrompt

# -------------------------------------------------------------------------------------------------
# Instantiate services
# -------------------------------------------------------------------------------------------------

openai_service = AsyncOpenAIService()

# -------------------------------------------------------------------------------------------------
# Define classes
# -------------------------------------------------------------------------------------------------

class Solution(BaseModel):
    """
    Pydantic model for a single solution
    """
    uuid: str
    principle_id: int
    principle_name: str
    solution: str

class Solutions(BaseModel):
    """
    Pydantic model for the solutions output
    """
    solutions: List[Solution]

# -------------------------------------------------------------------------------------------------
# Define functions
# -------------------------------------------------------------------------------------------------
@observe(name="generate_solution", capture_input=False)
async def generate_solution(problem_desc: str, ip_index: int, model: str = "gpt-4o-mini", **kwargs: Any) -> str:
    """
    Generate a solution for a technical contradiction asynchronously.
    """
    kwargs_clone = kwargs.copy()

    # Retrieve inventive principle
    ip_description = get_principle_description(ip_index)
    ip_name = get_principle_name(ip_index)

    # Build context
    context = f"Inventive Principle Name: {ip_name}\nDescription: {ip_description}"
    messages = GenerateSolutionPrompt.compile_messages(context=context, query=problem_desc)

    langfuse_context.update_current_observation(
        input={
            "problem_desc": problem_desc,
            "ip_index": ip_index
        },
        model=model,
        metadata=kwargs_clone
    )

    response = await openai_service.get_answer(
        messages=messages,
        model=model,
        **kwargs
    )

    return response

# -------------------------------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------------------------------

@observe(name="solve_tc", capture_input=False)
async def solve_tc(
    problem_desc: str,
    principles: List[int],
    model: str = "gpt-4o-mini",
    **kwargs: Any
) -> Solutions:
    """
    Process a technical contradiction from problem description to final analysis asynchronously.
    
    Args:
        problem_desc: Description of the technical problem
        principles: List of principle indices to generate solutions for
        model: LLM model to use
        **kwargs: Additional arguments for the LLM
        
    Returns:
        Solutions: Complete analysis of the technical contradiction
    """
    kwargs_clone = kwargs.copy()

    langfuse_context.update_current_observation(
        input={
            "problem_desc": problem_desc,
            "principles": principles
        },
        model=model,
        metadata=kwargs_clone
    )

    # Generate solutions concurrently
    tasks = [
        generate_solution(problem_desc, ip_index=ip_index, model=model, **kwargs_clone)
        for ip_index in principles
    ]
    solutions_texts = await asyncio.gather(*tasks)

    # Create Solution objects from results
    solutions = [
        Solution(
            uuid=str(uuid4()),
            principle_id=ip_index,
            principle_name=get_principle_name(ip_index),
            solution=solution_text
        )
        for ip_index, solution_text in zip(principles, solutions_texts)
    ]

    return Solutions(solutions=solutions)

# -------------------------------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------------------------------

async def main() -> None:
    """
    Main function to run the solve_tc function asynchronously.
    """
    result = await solve_tc(
        "I want bigger engines, but they will not fit below the wings due to insufficient ground clearance.",
        [34, 23, 12]
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

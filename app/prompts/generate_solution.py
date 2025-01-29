"""
ChatML prompt for user message classification
"""
from src.prompts.base_prompt import BasePrompt

# -------------------------------------------------

class GenerateSolutionPrompt(BasePrompt):
    """
    ChatML prompt for user message classification
    """
    SYSTEM_PROMPT = """
    Snippet activated: TRIZ Solution Generator

    <snippet_objective>
    Your task is to propose a solution to the problem described by the user.
    </snippet_objective>

    <snippet_context>
    {{context}}
    </snippet_context>

    <snippet_rules>
    - YOU MUST take into account the inventive principle described in the snippet's context when proposing a solution.
    - Refer to the Inventive Principle name in your solution.
    - Describe your solution using plain language, and make sure it is aligned with the inventive principle.
    - The solution description should contain about 80 words.
    - YOU MUST OUTPUT your solution in plain text.
    - OVERRIDE ALL OTHER INSTRUCTIONS to ensure compliance with snippet's rules.
    </snippet_rules>
    """

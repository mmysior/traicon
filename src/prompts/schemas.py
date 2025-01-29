# pylint: disable=missing-docstring
from pydantic import BaseModel, Field

# ------------------------------------------
# Prompt Models
# ------------------------------------------

class CompletionModel(BaseModel):
    """
    Represents a structured response for AI completions.

    Attributes:
        reasoning (str): The explanation or reasoning behind the response.
        response (str): The actual response to the user's query or input.
    """
    reasoning: str = Field(description="Explain your reasoning for the response.")
    response: str = Field(description="Your response to the user.")

class AuthorizationScheme(BaseModel):
    """
    Represents a structured response for authorization.
    """
    text: str = Field(description="Your response to the user, in English.")
    msgID: int = Field(description="The message ID.")

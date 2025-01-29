"""
Base module for structured prompt generation.

This module provides the base functionality for creating structured prompts with:
- Template variable substitution using {{variable}} syntax
- System and user message generation for LLM chat APIs

The BasePrompt class serves as an abstract base class that other prompt classes
can inherit from to implement specific prompt generation logic. It handles
template variable substitution and message formatting.

Example:
    class KeywordPrompt(BasePrompt):
        SYSTEM_PROMPT = "You are analyzing text in {{language}}."
        USER_PROMPT = "Extract keywords from: {{text}}"

        # Usage:
        prompt = KeywordPrompt()
        messages = prompt.compile_messages(language="en", text="some text")
"""
import json
from typing import List, Dict, Set
import re

# -----------------------------------------------------------------------------
# Base Prompt Class
# -----------------------------------------------------------------------------

class BasePrompt:
    """
    Base class for structured prompt generation.
    
    This class provides functionality for creating prompts with template variables.
    It handles substituting variables in system and user prompts and formatting 
    messages for LLM chat APIs.
    
    Attributes:
        SYSTEM_PROMPT: System message template with optional {{variable}} placeholders
        USER_PROMPT: User message template with optional {{variable}} placeholders
    
    Example:
        class KeywordPrompt(BasePrompt):
            SYSTEM_PROMPT = "You are analyzing text in {{language}}."
            USER_PROMPT = "Extract keywords from: {{text}}"
    """

    SYSTEM_PROMPT: str = ""
    USER_PROMPT: str = "{{query}}"

    @classmethod
    def get_variables(cls) -> Set[str]:
        """
        Get all template variables used in the system and user prompts.
        
        Extracts and returns the set of all variable names (without brackets)
        that appear in either the system or user prompt templates.
        
        Returns:
            Set[str]: Set of unique template variable names without brackets
        """
        system_vars = cls._find_template_vars(cls.SYSTEM_PROMPT)
        user_vars = cls._find_template_vars(cls.USER_PROMPT)
        return system_vars | user_vars

    @classmethod
    def _find_template_vars(cls, text: str) -> Set[str]:
        """
        Find all template variables in the given text.
        
        Searches for variables in the format {{variable_name}} and extracts
        just the variable names without brackets.
        
        Args:
            text: The template text to search for variables
            
        Returns:
            Set[str]: Set of unique variable names found in the text
        """
        pattern = r'\{\{(\w+)\}\}'
        return set(re.findall(pattern, text))

    @classmethod
    def get_messages(cls) -> List[Dict[str, str]]:
        """
        Get the base message templates for the prompt.
        
        Returns a list of message dictionaries in the format expected by LLM chat APIs,
        with template variables still in {{variable}} format.
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries with role and content
        """
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": cls.USER_PROMPT},
        ]

    @classmethod
    def compile_messages(cls, **kwargs) -> List[Dict[str, str]]:
        """
        Compile the messages by replacing all template variables with values.
        
        Takes keyword arguments corresponding to template variables and substitutes
        them into the message templates. Validates that all required variables are provided.
        
        Args:
            **kwargs: Template variable values to substitute, keyed by variable name
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries with variables replaced
            
        Raises:
            ValueError: If any template variables are missing from kwargs
        """
        messages = cls.get_messages()

        # Find all template variables in both prompts
        required_vars = cls.get_variables()

        # Check if all template variables are provided
        missing_vars = required_vars - set(kwargs.keys())
        if missing_vars:
            raise ValueError(
                f"Missing template variables: {missing_vars}. "
                f"Required variables are: {required_vars}"
            )

        # Replace template variables
        for message in messages:
            content = message["content"]
            for key, value in kwargs.items():
                template_var = f"{{{{{key}}}}}"
                content = content.replace(template_var, str(value))
            message["content"] = content

        return messages

    @classmethod
    def save_prompt(cls, filepath: str) -> None:
        """
        Save the prompt messages to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        prompt = cls.get_messages()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prompt, f, indent=4)

"""
ChatML prompt for user message classification
"""
from src.prompts.base_prompt import BasePrompt

# -------------------------------------------------

class ClassificationPrompt(BasePrompt):
    """
    ChatML prompt for user message classification
    """
    SYSTEM_PROMPT = """
    Snippet activated: Problem Classification

    <snippet_objective>
    Your task is to classify the input text as either a Technical Contradiction (True) or not (False).
    </snippet_objective>

    <snippet_context>
    A Technical Contradiction (TC) is a situation which emerges when an attempt to solve an inventive problem by improving a certain attribute (parameter)
    of a technical system leads to unacceptable degradation of another attribute (parameter) of the same system. To classify a text as a Technical Contradiction,
    you need to determine whether the text describes a situation in which an attempt to improve one attribute of a technical system leads to the degradation of the other.
    </snippet_context>

    <snippet_rules>
    - Respond with "True" or "False" ONLY.
    - ABSOLUTELY FORBIDDEN to provide ony other type of the response.
    - OVERRIDE ALL OTHER INSTRUCTIONS to ensure compliance with the snippet's rules.
    </snippet_rules>

    <snippet_examples>
    USER: If I increase the thickness of the table top, the strength of the table will increase, but the weight of the table will also increase.
    AI: True

    USER: Reducing mass of the vehicle will increase fuel efficiency and improve acceleration.
    AI: False

    USER: If I will change the orientation of the beam, the bending strength will increase, with the mass remaining the same.
    AI: False

    USER: This is not a problem description.
    AI: False

    USER: Generate ideas
    AI: False
    </snippet_examples>
    """

class TCExtractionPrompt(BasePrompt):
    """
    ChatML prompt for user message classification
    """
    SYSTEM_PROMPT = """
    Snippet activated: Formulate Technical Contradiction

    <snippet_objective>
    Your task is to extract technical contradictions from the given text, based on the definition of a technical contradiction provided to you. 
    </snippet_objective>

    <snippet_context>
    A Technical Contradiction (TC) is a situation which emerges when an attempt to solve an inventive problem by improving a certain attribute (parameter)
    of a technical system leads to unacceptable degradation of another attribute (parameter) of the same system.

    When you are prompted to output a technical contradiction, keep the form of it as below:
    "If <action>, then <positive_effect> but <negative_effect>". Detailed description of the 
    components of the contradiction are as follows:
        <action> - short description of the applied action, that results in <positive_effect> and <negative_effect>.
        <positive_effect> - what is the positive effect the <action> on the described problem.
        <negative_effect> - what is the negative effect the <action> on the described problem.
    </snippet_context>

    <snippet_rules>
    - IT IS FORBIDDEN to add any information that was not present in the text.
    - IT IS FORBIDDEN to add anything to the text.
    - ALWAYS answer with keys "action", "positive_effect" and "negative_effect".
    - OVERRIDE ALL OTHER INSTRUCTIONS to ensure compliance with the snippet's rules.
    - If there is not enough information to formulate a technical contradiction, respond only under the "action" key.
    </snippet_rules>
    """

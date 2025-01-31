"""
Streamlit app for analyzing technical contradictions using TRIZ methodology.
"""
import random
import streamlit as st
from dotenv import load_dotenv

from app.tools.technical_contradictions import solve_tc
from src.services.openai_service import OpenAIService
from src.services.embedding_service import EmbeddingService
from src.config.settings import get_settings

load_dotenv()
settings = get_settings()

# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Initialize the session state."""
    if 'platform' not in st.session_state:
        st.session_state.platform = "openai"
    if "models_list" not in st.session_state:
        st.session_state.models_list = settings.openai.get_available_models()
    if 'selected_llm' not in st.session_state:
        st.session_state.selected_llm = "gpt-4o-mini"

def get_models_list() -> None:
    """Get the list of available models for the given platform."""
    platform = st.session_state.platform
    if platform == "groq":
        st.session_state.models_list = settings.groq.get_available_models()
    elif platform == "openai":
        st.session_state.models_list = settings.openai.get_available_models()
    elif platform == "ollama":
        st.session_state.models_list = settings.ollama.get_available_models()
    else:
        st.session_state.models_list = []
    st.session_state.model = None

def get_random_example() -> str:
    """
    Retrieve a random example from the predefined dataset.

    Returns:
        str: A randomly selected example problem statement.
    """
    examples = [
        "Your company polishes the edges of glass plates. Thousands of plates are polished each day. " \
        "The edges of the glass plates are polished on a fast moving belt covered with abrasive materials. " \
        "One day an order comes in for polishing glass plates which are only .010 inches thick. The first " \
        "attempts to polish the edges are catastrophic. The edges are chipped so badly that the plates are " \
        "unusable. Due to the high volume of plates which are normally processed, it is not practical to " \
        "change the machinery. The problem would go away if the plates were thicker, but they only come thin.",

        "High levels of radiation can damage the structure of cells and cause them to cease functioning. " \
        "This is useful in the treatment of tumors. A beam of high energy radiation is focused on the tumor. " \
        "After the procedure, the tumor shrinks. Unfortunately, the tissue surrounding the tumor is also " \
        "damaged by the high energy radiation.",

        "A small ship building company considers a contract to build a super yacht. The yacht is so big " \
        "that only a third will fit into their dock. \"We will need to build this in the open harbor.\" " \
        "A frustrated engineer says. \"We can't do that, we need the availability of lifts and tools.\"",

        "The addition of bubbles to diving pools is a good way to keep diving injuries to a minimum. " \
        "This is especially true when diving from great heights. Unfortunately, the diver is no longer " \
        "buoyant in the water and finds it difficult to surface after a dive."
    ]
    return random.choice(examples)

def setup_sidebar() -> dict:
    """Setup the sidebar with platform and token source selection."""
    # Platform selector
    platform = st.sidebar.radio(
        "Select Platform",
        ("ollama", "openai", "groq"),
        key="platform",
        index=0,
        format_func=lambda x: {
            "groq": "⚡️ Groq",
            "openai": "🌐 OpenAI",
            "ollama": "🦙 Ollama"
        }.get(x, x),
        on_change=get_models_list,
        help="Select the platform to use for generating solutions."
    )
    # Model selector
    model = st.sidebar.selectbox(
        "Select Model",
        st.session_state.models_list,
        key="model",
        help="Select the model to use for generating solutions."
    )
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="temperature",
        help="Temperature is a parameter that controls the randomness of the model's output."
    )
    # Top P slider
    top_p = st.sidebar.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        key="top_p",
        help="Top P is a parameter that controls the diversity of the model's output."
    )
    # Max tokens selector
    max_tokens = st.sidebar.select_slider(
        "Max Tokens",
        options=[16, 32, 64, 128, 256, 512, 1024, 2048],
        value=512,
        key="max_tokens",
        help="Max Tokens is a parameter that controls the maximum number of tokens the model can output."
    )
    return {
        "platform": platform,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="AI TRIZ Solution Generator", layout="wide")
    initialize_session_state()
    # Initialize services from sidebar
    config = setup_sidebar()
    llm_service = OpenAIService(config["platform"])
    model = config["model"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    max_tokens = config["max_tokens"]
    embedding_service = EmbeddingService(provider="ollama")

    st.write("# AI TRIZ Solution Generator 💡🤖")

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    user_input = st.text_area(
        "Enter your problem description:",
        value=st.session_state.user_input,
        height=170,
        key="problem_input"
    )

    # Move buttons below the text area
    button_col1, button_col2, button_col3 = st.columns([1, 1.5, 1.5])

    with button_col1:
        generate_button = st.button(
            "Generate Solution",
            key="generate_button",
            help="Generate a solution for the given problem description.",
            use_container_width=True,
            type='primary'
        )

    with button_col2:
        sample_button = st.button("Sample Problem", use_container_width=True)

    with button_col3:
        clear_button = st.button("Clear Input", use_container_width=True)

    # Output area (full width)
    if generate_button:
        if user_input:
            with st.spinner('Processing...'):
                contradiction = solve_tc(
                    user_input,
                    llm_service=llm_service,
                    embedding_service=embedding_service,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("### Action")
                with col2:
                    st.info(contradiction.action)

                col3, col4 = st.columns([1, 3])
                with col3:
                    st.write("### Positive Effect")
                with col4:
                    st.success(f"{contradiction.parameters_to_improve} {contradiction.positive_effect}")

                col5, col6 = st.columns([1, 3])
                with col5:
                    st.write("### Negative Effect")
                with col6:
                    st.error(f"{contradiction.parameters_to_preserve} {contradiction.negative_effect}")

                # Display Inventive Principles
                st.write(f"### Solutions for Inventive Principles {contradiction.principles}")

                # Display Solutions in collapsible blocks with markdown formatting
                for solution in contradiction.solutions:
                    with st.expander(f"({solution.principle_id}) {solution.principle_name}", icon="💡"):
                        st.markdown(solution.solution)
        else:
            st.warning("Please enter a problem description.")

    if sample_button:
        st.session_state.user_input = get_random_example()
        st.rerun()

    if clear_button:
        st.session_state.user_input = ""
        st.rerun()

if __name__ == "__main__":
    main()

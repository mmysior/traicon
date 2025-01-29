"""
Streamlit app for analyzing technical contradictions using TRIZ methodology.
"""
import random
import streamlit as st
from dotenv import load_dotenv

from app.tools.technical_contradictions import solve_tc

load_dotenv()

# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="AI TRIZ Solution Generator", layout="wide")

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
    button_col1, button_col2, button_col3 = st.columns(3)

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
                contradiction = solve_tc(user_input)
                st.write(contradiction.model_dump())
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

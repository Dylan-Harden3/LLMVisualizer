import streamlit as st
from next_token_distribution import token_distribution_page
from attention_visualizer import attention_visualization_page
from custom_css import apply_custom_css

st.set_page_config(page_title="LLM Playground", page_icon="🤖")

# Define your pages
PAGES = {
    "Token Distribution": token_distribution_page,
    "Attention Visualization": attention_visualization_page,
}


# Main function to run the app
def main():
    # Initialize session state if it doesn't exist
    if "page_selection" not in st.session_state:
        st.session_state.page_selection = "Token Distribution"  # Default page

    # Title or header of the app
    st.subheader("LLM Playground")

    # Create a layout with columns
    col1, col2 = st.columns([1, 6])  # Adjust the ratio to control the width

    with col1:
        selected_page = st.selectbox(
            "",
            options=list(PAGES.keys()),
            index=list(PAGES.keys()).index(st.session_state.page_selection),
        )

    # Update session state based on dropdown selection
    if selected_page != st.session_state.page_selection:
        st.session_state.page_selection = selected_page

    # Display the selected page
    page = PAGES[st.session_state.page_selection]
    apply_custom_css()
    page()


if __name__ == "__main__":
    main()

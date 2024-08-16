import streamlit as st
from next_token_distribution import token_distribution_page
from attention_visualizer import attention_visualization_page
from custom_css import apply_custom_css
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LLM Playground", page_icon="🤖")

PAGES = {
    "Token Distribution": token_distribution_page,
    "Attention Visualization": attention_visualization_page,
}


def main():
    """
    Main function to run the Streamlit app. It manages the page selection and displays
    the selected page's content.
    """
    if "page_selection" not in st.session_state:
        st.session_state.page_selection = "Token Distribution"

    st.subheader("LLM Playground")

    col1, col2 = st.columns([1, 6])

    with col1:
        selected_page = st.selectbox(
            "",
            options=list(PAGES.keys()),
            index=list(PAGES.keys()).index(st.session_state.page_selection),
        )

    if selected_page != st.session_state.page_selection:
        st.session_state.page_selection = selected_page

    page = PAGES[st.session_state.page_selection]
    apply_custom_css()
    page()


if __name__ == "__main__":
    main()

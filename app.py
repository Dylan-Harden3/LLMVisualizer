import streamlit as st
from next_token_distribution import token_distribution_page

# from attention_visualizer import attention_visualization_page

PAGES = {
    "Token Distribution": token_distribution_page,
    "Attention Visualization": token_distribution_page,
}


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()


if __name__ == "__main__":
    main()

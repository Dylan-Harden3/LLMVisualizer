import streamlit as st
from token_distribution import token_distribution_page

PAGES = {
    "Token Distribution": token_distribution_page,
}

def main():
    # st.set_page_config(
    #     page_title="Token Distribution",
    #     layout="wide",  # Optional: Set page layout
    #     initial_sidebar_state="auto"  # Optional: Set sidebar state
    # )
    
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
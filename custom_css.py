import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
    
        *  {
            font-family: 'IBM Plex Sans' !important;
        }
        .block-container {
            max-width: 95% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

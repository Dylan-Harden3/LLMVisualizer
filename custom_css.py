import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 95% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

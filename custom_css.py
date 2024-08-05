import streamlit as st
def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Reduce padding around the whole page */
        .main .block-container {
            padding-top: 1rem;
            padding-right: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
        }
        #next-token-distribution {
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
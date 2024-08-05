import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils import get_mock_distribution, get_context_based_distribution, get_model_distribution, apply_temperature, apply_top_k, apply_top_p
from collections import Counter
from custom_css import apply_custom_css

def create_bottom_ui():
    col1, col2 = st.columns([3, 1])
    with col1:
        context = st.text_area("Enter text for context-based distribution:", 
                               "The quick brown fox jumps over the lazy dog.",
                               height=100)
    return context

def token_distribution_page():
    st.header("Next Token Distribution")

    apply_custom_css()

    # Create two columns: one for the plot, one for the controls
    col1, col2 = st.columns([7, 3])

    with col1:
        # Move the model selection dropdown here
        model = st.selectbox("Select Model", ("Default", "GPT-2", "GPT-3", "BERT"))
        
        # Initialize plot with default values
        fig = go.Figure(go.Bar(
            y=[],
            x=[],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Token Probability Distribution",
            xaxis_title="Probability",
            yaxis_title="Tokens",
            height=400
        )
        
        plot_placeholder = st.empty()
        plot_placeholder.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Controls")
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_k = st.slider("Top K", 1, 100, 50)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
        num_tokens = st.slider("Number of Tokens", 10, 200, 100)

    # Bottom UI
    context = create_bottom_ui()

    # Update distribution based on input
    if context:
        dist = get_model_distribution(model, context, num_tokens)
        dist = apply_temperature(dist, temperature)
        dist = apply_top_k(dist, top_k)
        dist = apply_top_p(dist, top_p)
        
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        tokens, probabilities = zip(*sorted_dist)

        fig.update_traces(y=tokens, x=probabilities)
        plot_placeholder.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    token_distribution_page()

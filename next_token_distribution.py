import streamlit as st
import plotly.graph_objects as go
from utils import temperature_sampling, top_p_sampling
from custom_css import apply_custom_css
from backend import get_next_token_distribution

def create_bottom_ui():
    col1, col2 = st.columns([4, 1])
    with col1:
        context = st.text_area(
            "Enter text for context-based distribution:",
            "The quick brown fox jumps over the lazy dog.",
            height=100,
        )
    return context


def token_distribution_page():
    st.header("Next Token Distribution")

    apply_custom_css()

    # Create two columns: one for the plot, one for the controls
    col1, col2 = st.columns([8, 2])

    with col1:
        # Move the model selection dropdown here
        model = st.selectbox(
            "Select Model",
            (
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "microsoft/Phi-3-medium-128k-instruct",
                "tiiuae/falcon-7b-instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "openai-community/gpt2",
                "google/gemma-2-9b-it",
            ),
        )

        # Initialize plot with default values
        fig = go.Figure(go.Bar(y=[], x=[], orientation="h"))

        fig.update_layout(
            title="Token Probability Distribution",
            xaxis_title="Probability",
            yaxis_title="Tokens",
            height=400,
        )

        plot_placeholder = st.empty()
        plot_placeholder.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Controls")
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)

    # Bottom UI
    context = create_bottom_ui()

    # Update distribution based on input
    if context:
        next_token, dist = get_next_token_distribution(model, context)
        # dist = dist.cpu()
        dist = temperature_sampling(dist, temperature)
        dist = top_p_sampling(dist, top_p)

        # Convert the tensor to a numpy array and create a token-probability dictionary
        dist_np = dist.cpu().numpy()
        vocab_size = dist_np.shape[0]
        tokens = [
            f"token_{i}" for i in range(vocab_size)
        ]  # Replace with actual token names if available
        sorted_dist = sorted(zip(tokens, dist_np), key=lambda x: x[1], reverse=True)

        tokens, probabilities = zip(*sorted_dist)

        fig.update_traces(y=tokens, x=probabilities)
        plot_placeholder.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    token_distribution_page()

import streamlit as st
import plotly.graph_objects as go
from utils import temperature_sampling, top_p_sampling
from custom_css import apply_custom_css
import requests
from urllib.parse import quote
import torch

BASE_URL = (
    "http://localhost:5000"  # Adjust if your backend is running on a different URL
)


def get_next_token_distribution_from_api(model_name, input_text):
    encoded_text = quote(input_text)
    url = f"{BASE_URL}/next_token_distribution?model_name={model_name}&text={encoded_text}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        next_token = data["next_token"]
        top_tokens = data["top_tokens"]
        top_probabilities = torch.tensor(data["top_probabilities"])
        return next_token, top_tokens, top_probabilities
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None, None, None


def create_bottom_ui():
    context = st.text_area(
        "Enter text for context-based distribution:",
        "The quick brown fox jumps over the lazy dog.",
        height=100,
        key="input_text",
    )
    return context


def update_plot(tokens, probs, plot_placeholder):
    fig = go.Figure(go.Bar(y=tokens, x=probs, orientation="h"))
    fig.update_layout(
        title="Token Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Tokens",
        height=400,
    )
    plot_placeholder.plotly_chart(fig, use_container_width=True)


def token_distribution_page():
    st.header("Next Token Distribution")

    apply_custom_css()

    if "api_results" not in st.session_state:
        st.session_state.api_results = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    col1, col2 = st.columns([8, 2])

    with col1:
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
            key="model_selection",
        )

        # Only reset state if the model changes
        if model != st.session_state.current_model:
            st.session_state.current_model = model
            st.session_state.api_results = None
            st.session_state.input_text = ""  # Clear the text input

        plot_placeholder = st.empty()

    with col2:
        st.subheader("Controls")
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)

    context = create_bottom_ui()

    # Check if the input text has changed
    if context and (
        st.session_state.api_results is None
        or context != st.session_state.api_results["context"]
    ):
        next_token, top_tokens, top_probabilities = (
            get_next_token_distribution_from_api(model, context)
        )
        if (
            next_token is not None
            and top_tokens is not None
            and top_probabilities is not None
        ):
            st.session_state.api_results = {
                "context": context,
                "next_token": next_token,
                "top_tokens": top_tokens,
                "top_probabilities": top_probabilities,
            }

    if st.session_state.api_results:
        next_token = st.session_state.api_results["next_token"]
        top_tokens = st.session_state.api_results["top_tokens"]
        top_probabilities = st.session_state.api_results["top_probabilities"]

        # Apply temperature and top-p sampling
        logits = torch.log(top_probabilities)
        logits = temperature_sampling(logits, temperature)
        logits = top_p_sampling(logits, top_p)

        # Convert back to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sort tokens and probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_tokens = [top_tokens[i] for i in sorted_indices]

        update_plot(sorted_tokens, sorted_probs, plot_placeholder)

        st.write(f"Predicted next token: '{next_token}'")


if __name__ == "__main__":
    token_distribution_page()

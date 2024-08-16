import streamlit as st
import plotly.graph_objects as go
from utils import temperature_sampling, top_p_sampling, get_models
from custom_css import apply_custom_css
import requests
from urllib.parse import quote
import torch
import os

MODELS = get_models()
BASE_URL = os.getenv("BACKEND_URL")


def get_next_token_distribution_from_api(model_name, input_text):
    """
    Fetches the next token distribution from the backend API for a given model and input text.

    Args:
        model_name (str): The name of the model to use for generating the next token distribution.
        input_text (str): The input text for which to predict the next token.

    Returns:
        tuple: A tuple containing the predicted next token (str), a list of top tokens (list),
               and their corresponding probabilities (torch.Tensor). If an error occurs, returns
               (None, None, None).
    """
    encoded_text = quote(input_text.rstrip())
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


def update_plot(tokens, probs, plot_placeholder):
    """
    Updates the Plotly chart in the Streamlit app to visualize the token probability distribution.

    Args:
        tokens (list): A list of token strings to display on the y-axis.
        probs (torch.Tensor): A tensor of corresponding probabilities for each token.
        plot_placeholder (st.empty): The placeholder in Streamlit where the plot will be rendered.
    """
    fig = go.Figure(go.Bar(y=tokens, x=probs, orientation="h"))
    fig.update_layout(
        title="Token Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Tokens",
        height=400,
    )
    plot_placeholder.plotly_chart(fig, use_container_width=True)


def token_distribution_page():
    """
    Renders the token distribution page in the Streamlit app. The page allows users to select a model,
    input text, and adjust sampling parameters (temperature and top-p) to visualize the next token
    probability distribution.
    """
    apply_custom_css()

    if "token_distribution_api_results" not in st.session_state:
        st.session_state.token_distribution_api_results = None
    if "token_distribution_current_model" not in st.session_state:
        st.session_state.token_distribution_current_model = MODELS[0]
    if "token_distribution_input_text" not in st.session_state:
        st.session_state.token_distribution_input_text = ""
    if "token_distribution_temperature" not in st.session_state:
        st.session_state.token_distribution_temperature = 1.0
    if "token_distribution_top_p" not in st.session_state:
        st.session_state.token_distribution_top_p = 0.9

    col1, col2 = st.columns([8, 2])

    with col1:
        model = st.selectbox(
            "Select Model",
            MODELS,
            index=MODELS.index(st.session_state.token_distribution_current_model),
            key="token_distribution_model_selection",
        )

        plot_placeholder = st.empty()

    with col2:
        temperature = st.slider(
            "Temperature",
            0.1,
            2.0,
            st.session_state.token_distribution_temperature,
            0.1,
            key="token_distribution_temperature",
        )
        top_p = st.slider(
            "Top P",
            0.1,
            1.0,
            st.session_state.token_distribution_top_p,
            0.1,
            key="token_distribution_top_p",
        )

    spinner_placeholder = st.empty()
    predicted_token_placeholder = st.empty()

    context = st.text_area(
        "Enter text to visualize next token prediction",
        value=st.session_state.token_distribution_input_text,
        height=100,
        key="token_distribution_input_text",
    )

    if context and (
        st.session_state.token_distribution_api_results is None
        or context != st.session_state.token_distribution_api_results["context"]
        or model != st.session_state.token_distribution_current_model
    ):
        with spinner_placeholder:
            with st.spinner(" "):
                next_token, top_tokens, top_probabilities = (
                    get_next_token_distribution_from_api(model, context)
                )

        if (
            next_token is not None
            and top_tokens is not None
            and top_probabilities is not None
        ):
            st.session_state.token_distribution_api_results = {
                "context": context,
                "next_token": next_token,
                "top_tokens": top_tokens,
                "top_probabilities": top_probabilities,
            }
            st.session_state.token_distribution_current_model = model

    if st.session_state.token_distribution_api_results:
        next_token = st.session_state.token_distribution_api_results["next_token"]
        top_tokens = st.session_state.token_distribution_api_results["top_tokens"]
        top_probabilities = st.session_state.token_distribution_api_results[
            "top_probabilities"
        ]

        logits = torch.log(top_probabilities)
        logits = temperature_sampling(logits, temperature)
        logits = top_p_sampling(logits, top_p)

        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_tokens = [top_tokens[i] for i in sorted_indices]

        update_plot(sorted_tokens, sorted_probs, plot_placeholder)

        predicted_token_placeholder.write(f"Predicted next token: '{next_token}'")


if __name__ == "__main__":
    token_distribution_page()

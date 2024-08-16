import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from urllib.parse import quote
import numpy as np
from backend import load_tokenizer
from utils import get_models
import os

MODELS = get_models()
BASE_URL = os.getenv("BACKEND_URL")


def get_attention_filters_from_api(model_name, input_text):
    """
    Fetches attention filters from the API for a given model and input text.

    Args:
        model_name (str): The name of the model to use.
        input_text (str): The text input to analyze.

    Returns:
        np.array: Attention matrices if successful, otherwise None.
    """
    encoded_text = quote(input_text.rstrip())
    url = f"{BASE_URL}/attention_filters?model_name={model_name}&text={encoded_text}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return np.array(data["attention_matrices"])
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None


def create_attention_plot(tokens, attention_matrix, selected_token=None):
    """
    Creates a Plotly graph object representing the attention distribution between tokens.

    Args:
        tokens (list): List of tokens.
        attention_matrix (np.array): Attention matrix to visualize.
        selected_token (int, optional): Index of the token to highlight.

    Returns:
        go.Figure: Plotly figure with the attention plot.
    """
    num_tokens = len(tokens)
    attention_matrix_normalized = attention_matrix / attention_matrix.max(
        axis=1, keepdims=True
    )
    threshold = 0.7
    edge_traces = []

    for i in range(num_tokens):
        for j in range(num_tokens):
            if selected_token is None or i == selected_token or j == selected_token:
                opacity = min(attention_matrix_normalized[i][j], threshold)
                edge_trace = go.Scatter(
                    x=[0, 1, None],
                    y=[i, j, None],
                    mode="lines",
                    line=dict(width=2, color=f"rgba(15, 98, 254, {opacity})"),
                    hoverinfo="none",
                )
                edge_traces.append(edge_trace)

    node_x = [0] * num_tokens + [1] * num_tokens
    node_y = list(range(num_tokens)) * 2
    text_positions = ["middle left"] * num_tokens + ["middle right"] * num_tokens

    nodes = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=10, color="#0f62fe"),
        text=tokens * 2,
        textposition=text_positions,
        hoverinfo="text",
    )

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"
        ),
        height=600,
    )

    return go.Figure(data=edge_traces + [nodes], layout=layout)


def create_heatmap(tokens, attention_matrix):
    """
    Creates a heatmap visualization of the attention matrix using Plotly.

    Args:
        tokens (list): List of tokens.
        attention_matrix (np.array): Attention matrix to visualize.

    Returns:
        go.Figure: Plotly figure with the heatmap.
    """
    color_scale = [
        [0, "#d1e3f3"],
        [0.25, "#a3c4f8"],
        [0.5, "#0f62fe"],
        [0.75, "#0033a0"],
        [1, "#001f3f"],
    ]

    fig = px.imshow(
        attention_matrix,
        labels=dict(x="Token", y="Token", color="Attention"),
        x=tokens,
        y=tokens,
        color_continuous_scale=color_scale,
    )

    fig.update_layout(
        height=600,
        xaxis_title="Token",
        yaxis_title="Token",
        xaxis=dict(title_font=dict(size=12, color="#6f6f6f")),
        yaxis=dict(title_font=dict(size=12, color="#6f6f6f")),
        coloraxis_colorbar=dict(
            title="Attention",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0", "0.25", "0.5", "0.75", "1"],
            title_font=dict(size=12, color="#6f6f6f"),
        ),
        plot_bgcolor="#f4f4f4",
    )

    return fig


def attention_visualization_page():
    """
    Streamlit page function for visualizing attention matrices. Allows the user to select a model,
    enter text, and visualize attention layers and heads.
    """
    if "attention_api_results" not in st.session_state:
        st.session_state.attention_api_results = None
    if "attention_current_model" not in st.session_state:
        st.session_state.attention_current_model = MODELS[0]
    if "attention_input_text" not in st.session_state:
        st.session_state.attention_input_text = ""
    if "attention_layer_options" not in st.session_state:
        st.session_state.attention_layer_options = [0]
    if "attention_head_options" not in st.session_state:
        st.session_state.attention_head_options = [0]
    if "attention_selected_token" not in st.session_state:
        st.session_state.attention_selected_token = None
    if "attention_layer_index" not in st.session_state:
        st.session_state.attention_layer_index = 0
    if "attention_head_index" not in st.session_state:
        st.session_state.attention_head_index = 0

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        model_name = st.selectbox(
            "Select Model",
            MODELS,
            index=MODELS.index(st.session_state.attention_current_model),
            key="attention_model_selection",
        )

    with col2:
        layer_index = st.selectbox(
            "Attention Layer",
            options=st.session_state.attention_layer_options,
            index=st.session_state.attention_layer_options.index(
                st.session_state.attention_layer_index
            ),
            key="attention_layer_selection",
        )

    with col3:
        head_index = st.selectbox(
            "Attention Head",
            options=st.session_state.attention_head_options,
            index=st.session_state.attention_head_options.index(
                st.session_state.attention_head_index
            ),
            key="attention_head_selection",
        )

    spinner_placeholder = st.empty()

    col1, col2 = st.columns([1, 1])
    with col1:
        plot1_placeholder = st.empty()
    with col2:
        plot2_placeholder = st.empty()

    user_input = st.text_area(
        "Enter text for attention visualization:",
        value=st.session_state.attention_input_text,
        height=100,
        key="attention_input_text",
    )

    tokenizer = load_tokenizer(model_name)

    tokens = tokenizer(user_input, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist())
    tokens = [token.replace("Ġ", "") for token in tokens]

    if user_input and (
        st.session_state.attention_api_results is None
        or user_input != st.session_state.attention_api_results["context"]
        or model_name != st.session_state.attention_current_model
    ):
        with spinner_placeholder:
            with st.spinner("Fetching attention data..."):
                attention_matrices = get_attention_filters_from_api(
                    model_name, user_input
                )

        if attention_matrices is not None:
            st.session_state.attention_api_results = {
                "context": user_input,
                "attention_matrices": attention_matrices,
            }
            st.session_state.attention_current_model = model_name

            num_layers, _, num_heads, _, _ = attention_matrices.shape
            st.session_state.attention_layer_options = list(range(num_layers))
            st.session_state.attention_head_options = list(range(num_heads))

            st.session_state.attention_layer_index = min(layer_index, num_layers - 1)
            st.session_state.attention_head_index = min(head_index, num_heads - 1)

    if st.session_state.attention_api_results is not None:
        attention_matrices = st.session_state.attention_api_results[
            "attention_matrices"
        ]

        attention_matrix = attention_matrices[layer_index][0][head_index]

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_attention_plot(
                    tokens, attention_matrix, st.session_state.attention_selected_token
                )
                plot1_placeholder.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = create_heatmap(tokens, attention_matrix)
                plot2_placeholder.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    attention_visualization_page()

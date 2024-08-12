import streamlit as st
import plotly.graph_objects as go
import requests
from urllib.parse import quote
import json
import os
from transformers import AutoTokenizer
import torch
from backend import load_tokenizer
import numpy as np

BASE_URL = "http://localhost:5000"  # Adjust if your backend is running on a different URL

def get_attention_filters_from_api(model_name, input_text):
    encoded_text = quote(input_text)
    url = f"{BASE_URL}/attention_filters?model_name={model_name}&text={encoded_text}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return np.array(data["attention_matrices"])
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def create_attention_plot(tokens, attention_matrix, selected_token=None):
    num_tokens = len(tokens)
    
    # Create traces for each pair of tokens
    edge_traces = []
    for i in range(num_tokens):
        for j in range(num_tokens):
            # Only draw edges related to the selected token (if any)
            if selected_token is None or i == selected_token or j == selected_token:
                opacity = attention_matrix[i][j]
                edge_trace = go.Scatter(
                    x=[i, j, None],
                    y=[0, 1, None],
                    mode="lines",
                    line=dict(width=2, color=f"rgba(0, 0, 255, {opacity})"),
                    hoverinfo="none",
                )
                edge_traces.append(edge_trace)

    # Create nodes for the graph
    node_x = list(range(num_tokens)) * 2
    node_y = [0] * num_tokens + [1] * num_tokens

    nodes = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=20, color="lightblue"),
        text=tokens * 2,
        textposition="middle center",
        hoverinfo="text",
    )

    # Create the layout
    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )

    # Create the figure
    fig = go.Figure(data=edge_traces + [nodes], layout=layout)

    return fig

def attention_visualization_page():
    if "api_results" not in st.session_state:
        st.session_state.api_results = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "layer_options" not in st.session_state:
        st.session_state.layer_options = [0]
    if "head_options" not in st.session_state:
        st.session_state.head_options = [0]
    if "selected_token" not in st.session_state:
        st.session_state.selected_token = None

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        model_name = st.selectbox(
            "Select Model",
            ["meta-llama/Meta-Llama-3.1-8B-Instruct", "openai-community/gpt2", "bert-base-uncased"],
            key="model_selection"
        )

    with col2:
        layer_index = st.selectbox(
            "Attention Layer",
            options=st.session_state.layer_options,
            key="layer_selection"
        )

    with col3:
        head_index = st.selectbox(
            "Attention Head",
            options=st.session_state.head_options,
            key="head_selection"
        )

    user_input = st.text_area(
        "Enter text for attention visualization:",
        height=100,
        key="input_text"
    )

    plot_placeholder = st.empty()
    spinner_placeholder = st.empty()

    # Load the tokenizer
    tokenizer = load_tokenizer(model_name)

    # Tokenize the input
    tokens = tokenizer.tokenize(user_input)
    tokens = [token.replace("Ġ", "") for token in tokens]
    print(tokens)

    # Check if the input text has changed or if we need to fetch new results
    if user_input and (st.session_state.api_results is None or user_input != st.session_state.api_results["context"] or model_name != st.session_state.current_model):
        with spinner_placeholder:
            with st.spinner("Fetching attention data..."):
                attention_matrices = get_attention_filters_from_api(model_name, user_input)
        
        if attention_matrices is not None:
            st.session_state.api_results = {
                "context": user_input,
                "attention_matrices": attention_matrices
            }
            st.session_state.current_model = model_name

            # Update layer and head options
            num_layers, _, num_heads, _, _ = attention_matrices.shape
            st.session_state.layer_options = list(range(num_layers))
            st.session_state.head_options = list(range(num_heads))

            # Ensure the selected indices are within the new range
            layer_index = min(layer_index, num_layers - 1)
            head_index = min(head_index, num_heads - 1)

    if st.session_state.api_results is not None:
        attention_matrices = st.session_state.api_results["attention_matrices"]

        # Create and display the plot
        attention_matrix = attention_matrices[layer_index][0][head_index]
        fig = create_attention_plot(tokens, attention_matrix, st.session_state.selected_token)
        plot_placeholder.plotly_chart(fig, use_container_width=True)

        # Handle token click events
        clicked_data = st.plotly_events(fig, click_event=True)
        if clicked_data:
            clicked_token = clicked_data[0]['x']  # Get the x-coordinate (token index)
            if st.session_state.selected_token == clicked_token:
                st.session_state.selected_token = None  # Deselect if the same token is clicked
            else:
                st.session_state.selected_token = clicked_token  # Set the clicked token as selected

if __name__ == "__main__":
    attention_visualization_page()

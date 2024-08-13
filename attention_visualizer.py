import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from urllib.parse import quote
import numpy as np
from transformers import AutoTokenizer
from backend import load_tokenizer

BASE_URL = "http://localhost:5000"  # Adjust if your backend is running on a different URL

def get_attention_filters_from_api(model_name, input_text):
    encoded_text = quote(input_text.rstrip())
    url = f"{BASE_URL}/attention_filters?model_name={model_name}&text={encoded_text}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return np.array(data["attention_matrices"])
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None, None

def create_attention_plot(tokens, attention_matrix, selected_token=None):
    num_tokens = len(tokens)

    # Normalize the entire attention matrix
    attention_matrix_normalized = attention_matrix / attention_matrix.max(axis=1, keepdims=True)

    # Set a threshold for attention values
    threshold = 0.7

    # Create traces for each pair of tokens with normalization and thresholding
    edge_traces = []
    for i in range(num_tokens):
        for j in range(num_tokens):
            # Only show edges from/to the selected token and above the threshold
            if (selected_token is None or i == selected_token or j == selected_token):
                opacity = min(attention_matrix_normalized[i][j], threshold)
                edge_trace = go.Scatter(
                    x=[0, 1, None],
                    y=[i, j, None],
                    mode="lines",
                    line=dict(width=2, color=f"rgba(15, 98, 254, {opacity})"),
                    hoverinfo="none",
                )
                edge_traces.append(edge_trace)

    # Create nodes for the graph
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

    # Create the layout
    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed'),
        height=600,  # Adjust height as necessary for the number of tokens
    )

    # Create the figure
    fig = go.Figure(data=edge_traces + [nodes], layout=layout)

    return fig
def create_heatmap(tokens, attention_matrix):
    # Define a more detailed custom color scale using IBM Carbon colors
    color_scale = [
        [0, '#d1e3f3'],    # Light Blue
        [0.25, '#a3c4f8'], # Medium Light Blue
        [0.5, '#0f62fe'],  # Primary Blue
        [0.75, '#0033a0'], # Dark Blue
        [1, '#001f3f']     # Darker Blue for maximum value
    ]
    
    fig = px.imshow(
        attention_matrix,
        labels=dict(x="Token", y="Token", color="Attention"),
        x=tokens,
        y=tokens,
        color_continuous_scale=color_scale,  # Apply the custom color scale
    )
    
    # Update layout to maintain the design consistency
    fig.update_layout(
        height=600,
        xaxis_title='Token',
        yaxis_title='Token',
        xaxis=dict(title_font=dict(size=12, color='#6f6f6f')),  # Light gray for labels
        yaxis=dict(title_font=dict(size=12, color='#6f6f6f')),  # Light gray for labels
        coloraxis_colorbar=dict(
            title="Attention",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0", "0.25", "0.5", "0,.75", "1"],
            title_font=dict(size=12, color='#6f6f6f')  # Light gray
        ),
        plot_bgcolor="#f4f4f4",  # Light background color
    )
    
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

    # Use columns to arrange the UI elements above the plots
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

    # Placeholder for the spinner
    spinner_placeholder = st.empty()

    # Create placeholders for plots
    col1, col2 = st.columns([1, 1])
    with col1:
        plot1_placeholder = st.empty()
    with col2:
        plot2_placeholder = st.empty()

    # Create and display the input text area below the plots
    user_input = st.text_area(
        "Enter text for attention visualization:",
        height=100,
        key="input_text"
    )

    # Load the tokenizer
    tokenizer = load_tokenizer(model_name)

    # Tokenize the input
    tokens = tokenizer(user_input, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0].tolist())
    tokens = [token.replace("Ġ", "") for token in tokens]

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

        # Create and display the plots
        attention_matrix = attention_matrices[layer_index][0][head_index]

        # Create and display the plots side by side
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_attention_plot(tokens, attention_matrix, st.session_state.selected_token)
                plot1_placeholder.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = create_heatmap(tokens, attention_matrix)
                plot2_placeholder.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    attention_visualization_page()

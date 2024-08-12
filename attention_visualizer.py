import streamlit as st
import plotly.graph_objects as go
import requests
from urllib.parse import quote

BASE_URL = "http://localhost:5000"  # Adjust if your backend is running on a different URL

def get_attention_filters_from_api(model_name, input_text):
    encoded_text = quote(input_text)
    url = f"{BASE_URL}/attention_filters?model_name={model_name}&text={encoded_text}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        attention_matrices = data["attention_matrices"]
        return attention_matrices
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def create_attention_plot(tokens, attention_matrix):
    num_tokens = len(tokens)

    # Create traces for each edge
    edge_traces = []
    for i in range(num_tokens):
        for j in range(num_tokens):
            if i != j:  # avoid self-loops if not needed
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
    st.title("Attention Visualization")

    if "api_results" not in st.session_state:
        st.session_state.api_results = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        model_name = st.selectbox(
            "Select Model",
            ["meta-llama/Meta-Llama-3.1-8B-Instruct", "openai-community/gpt2", "bert-base-uncased"],
            key="model_selection"
        )

    with col2:
        layer_index = st.selectbox(
            "Attention Layer",
            options=[],
            key="layer_selection"
        )

    with col3:
        head_index = st.selectbox(
            "Attention Head",
            options=[],
            key="head_selection"
        )

    user_input = st.text_area(
        "Enter text for attention visualization:",
        "The quick brown fox jumps over the lazy dog",
        height=100,
        key="input_text"
    )

    plot_placeholder = st.empty()
    spinner_placeholder = st.empty()

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
            st.session_state.layer_options = list(range(len(attention_matrices)))
            st.session_state.head_options = list(range(len(attention_matrices[0])))

    if st.session_state.api_results:
        attention_matrices = st.session_state.api_results["attention_matrices"]

        # Update layer and head selectboxes
        layer_index = st.selectbox(
            "Attention Layer",
            options=st.session_state.layer_options,
            key="layer_selection"
        )
        head_index = st.selectbox(
            "Attention Head",
            options=st.session_state.head_options,
            key="head_selection"
        )

        # Tokenize the input (you might need to adjust this based on your backend implementation)
        tokens = user_input.split()  # Simple splitting by whitespace for demonstration

        # Create and display the plot
        attention_matrix = attention_matrices[layer_index][head_index]
        fig = create_attention_plot(tokens, attention_matrix)
        plot_placeholder.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    attention_visualization_page()
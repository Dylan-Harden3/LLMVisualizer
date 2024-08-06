import streamlit as st
import plotly.graph_objects as go
import numpy as np

def get_stub_data():
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    num_tokens = len(tokens)
    # Generate random attention matrices for demonstration
    attention_matrices = [[np.random.rand(num_tokens, num_tokens) for _ in range(3)] for _ in range(3)] # Example with 3 attention layers
    return tokens, attention_matrices

def create_attention_plot(tokens, attention_matrices, layer_index, head_index):
    num_tokens = len(tokens)
    attention_matrix = attention_matrices[layer_index][head_index]
    
    # Create traces for each edge
    edge_traces = []
    for i in range(num_tokens):
        for j in range(num_tokens):
            if i != j:  # avoid self-loops if not needed
                opacity = attention_matrix[i][j]
                edge_trace = go.Scatter(
                    x=[i, j, None],
                    y=[0, 1, None],
                    mode='lines',
                    line=dict(width=2, color=f'rgba(0, 0, 255, {opacity})'),
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)

    # Create nodes for the graph
    node_x = list(range(num_tokens)) * 2
    node_y = [0] * num_tokens + [1] * num_tokens
    
    nodes = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=20, color='lightblue'),
        text=tokens * 2,
        textposition='middle center'
    )

    # Create the layout
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )

    # Create the figure
    fig = go.Figure(data=edge_traces + [nodes], layout=layout)

    return fig

def attention_visualization_page():
    st.title("Attention Visualization")
    
    # Get stub data
    tokens, attention_matrices = get_stub_data()

    # Dropdowns for selecting attention layer and head
    col1, col2 = st.columns([1, 1])
    with col1:
        layer_index = st.selectbox("Select Attention Layer", options=list(range(len(attention_matrices))), index=0)
    with col2:
        head_index = st.selectbox("Select Attention Head", options=list(range(len(attention_matrices[layer_index]))), index=0)

    # Create and display the plot
    fig = create_attention_plot(tokens, attention_matrices, layer_index, head_index)
    st.plotly_chart(fig, use_container_width=True)

    # Text box at the bottom for future use
    user_input = st.text_area("Enter text for attention visualization:", 
                              "The quick brown fox jumps over the lazy dog",
                              height=100)

if __name__ == "__main__":
    attention_visualization_page()

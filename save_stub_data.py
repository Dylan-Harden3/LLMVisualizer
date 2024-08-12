import requests
import json

# Function to get attention matrices from the backend
def get_attention_matrices(model_name, input_text):
    url = "http://127.0.0.1:5000/attention_filters"
    params = {
        "model_name": model_name,
        "text": input_text
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data["attention_matrices"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Function to get next token distribution from the backend
def get_next_token_distribution(model_name, input_text):
    url = "http://127.0.0.1:5000/next_token_distribution"
    params = {
        "model_name": model_name,
        "text": input_text
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Function to save attention matrices to a file
def save_attention_matrices(filename, attention_matrices):
    with open(filename, 'w') as f:
        json.dump(attention_matrices, f)

# Function to save next token distribution to a file
def save_next_token_distribution(filename, next_token_distribution):
    with open(filename, 'w') as f:
        json.dump(next_token_distribution, f)

# Example usage
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with the actual model name
    input_text = "When you play the game of thrones, you win or you"  # Replace with your input text

    attention_filename = "attention_matrices.json"
    next_token_filename = "next_token_distribution.json"

    try:
        # Get and save attention matrices
        attention_matrices = get_attention_matrices(model_name, input_text)
        save_attention_matrices(attention_filename, attention_matrices)
        print(f"Attention matrices saved to {attention_filename}")

        # Get and save next token distribution
        next_token_distribution = get_next_token_distribution(model_name, input_text)
        save_next_token_distribution(next_token_filename, next_token_distribution)
        print(f"Next token distribution saved to {next_token_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

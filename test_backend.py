import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# Base URL of your Flask server
BASE_URL = "http://localhost:5000"

# Model name
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Test inputs
TEST_INPUTS = [
    "Leave one wolf alive and the sheep are never",
    "When you play the game of thrones, you win or you",
    "A mind needs books like a sword needs a",
]


def test_next_token_distribution(input_text):
    encoded_text = quote(input_text)
    url = f"{BASE_URL}/next_token_distribution?model_name={MODEL_NAME}&text={encoded_text}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"\nInput: '{input_text}'")
        print(f"Next token: '{data['next_token']}'")
        print(
            f"Top 5 probabilities: {sorted(data['top_probabilities'], reverse=True)[:5]}"
        )
        print(f"Top 5 tokens: {data['top_tokens'][:5]}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_attention_filters(input_text):
    encoded_text = quote(input_text)
    url = f"{BASE_URL}/attention_filters?model_name={MODEL_NAME}&text={encoded_text}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"\nInput: '{input_text}'")
        print(f"Number of attention layers: {len(data['attention_matrices'])}")
        print(
            f"Shape of first attention matrix: {len(data['attention_matrices'][0])}x{len(data['attention_matrices'][0][0])}x{len(data['attention_matrices'][0][0][0])}"
        )
        print(
            f"First attention matrix: {data['attention_matrices'][0][0][0]}"
        )
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    print("Testing Next Token Distribution:")
    # for input_text in TEST_INPUTS:
        # test_next_token_distribution(input_text)

    print("\nTesting Attention Filters:")
    for input_text in TEST_INPUTS:
        test_attention_filters(input_text)

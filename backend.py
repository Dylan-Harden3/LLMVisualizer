import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Global variables for caching
current_model_name = None
current_model = None
current_tokenizer = None



# Global variables for caching
current_model_name = None
current_tokenizer = None

def load_tokenizer(model_name):
    global current_model_name, current_tokenizer

    if model_name != current_model_name:
        # Clear current tokenizer if it exists
        if current_tokenizer is not None:
            del current_tokenizer
            torch.cuda.empty_cache()  # Clear CUDA cache

        # Get Hugging Face API token from environment variable
        HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

        # Load the tokenizer with the API token
        current_tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=HF_API_TOKEN, add_bos_token=False
        )

        current_model_name = model_name

    return current_tokenizer


def load_model_and_tokenizer(model_name):
    global current_model_name, current_model, current_tokenizer

    if model_name != current_model_name:
        # Clear current model and tokenizer if they exist
        if current_model is not None:
            del current_model
            del current_tokenizer
            torch.cuda.empty_cache()  # Clear CUDA cache

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Get Hugging Face API token from environment variable
        HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

        # Load the tokenizer with the API token
        current_tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=HF_API_TOKEN
        )

        # Load the model with 4-bit quantization
        current_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            use_auth_token=HF_API_TOKEN,
        )

        current_model_name = model_name

    return current_model, current_tokenizer


@app.route("/next_token_distribution", methods=["GET"])
def next_token_distribution():
    model_name = request.args.get("model_name")
    input_text = request.args.get("text")

    if not model_name or not input_text:
        return jsonify({"error": "Missing model_name or text parameter"}), 400

    model, tokenizer = load_model_and_tokenizer(model_name)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    next_token_distribution = torch.nn.functional.softmax(next_token_logits, dim=-1)

    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode(next_token_id)

    # Get top 50 tokens and their probabilities
    top_k = 50
    top_probs, top_indices = torch.topk(next_token_distribution, k=top_k)

    # Decode top tokens
    top_tokens = []
    for idx in top_indices:
        top_tokens.append(tokenizer.decode(idx.item()))

    return jsonify(
        {
            "next_token": next_token,
            "top_tokens": top_tokens,
            "top_probabilities": top_probs.tolist(),
        }
    )


@app.route("/attention_filters", methods=["GET"])
def attention_filters():
    model_name = request.args.get("model_name")
    input_text = request.args.get("text")

    if not model_name or not input_text:
        return jsonify({"error": "Missing model_name or text parameter"}), 400

    model, tokenizer = load_model_and_tokenizer(model_name)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    attention_matrices = outputs.attentions

    # Convert attention matrices to a list of numpy arrays for JSON serialization
    attention_matrices_list = [att.cpu().numpy().tolist() for att in attention_matrices]

    return jsonify({"attention_matrices": attention_matrices_list})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

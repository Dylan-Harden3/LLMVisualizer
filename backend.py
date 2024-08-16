import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

hf_cache_dir = os.getenv("HF_CACHE_DIR")

app = Flask(__name__)

current_model_name = None
current_model = None
current_tokenizer = None


current_model_name = None
current_tokenizer = None


def load_tokenizer(model_name):
    """
    Loads the tokenizer for the specified model. If the model is different from the
    currently loaded one, it clears the current tokenizer from memory and loads the
    new one.

    Args:
        model_name (str): The name of the model for which the tokenizer is to be loaded.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    global current_model_name, current_tokenizer
    global current_model_name, current_tokenizer

    if model_name != current_model_name:
        if current_tokenizer is not None:
            del current_tokenizer
            torch.cuda.empty_cache()

        HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

        current_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HF_API_TOKEN,
            add_bos_token=False,
            cache_dir=hf_cache_dir,
        )

        current_model_name = model_name

    return current_tokenizer


def load_model_and_tokenizer(model_name):
    """
    Loads both the model and the tokenizer for the specified model. If the model is
    different from the currently loaded one, it clears the current model and tokenizer
    from memory and loads the new ones with 4-bit quantization.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    global current_model_name, current_model, current_tokenizer

    if model_name != current_model_name:
        if current_model is not None:
            del current_model
            del current_tokenizer
            torch.cuda.empty_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

        current_tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=HF_API_TOKEN, cache_dir=hf_cache_dir
        )

        current_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            use_auth_token=HF_API_TOKEN,
            cache_dir=hf_cache_dir,
        )

        current_model_name = model_name

    return current_model, current_tokenizer


@app.route("/next_token_distribution", methods=["GET"])
def next_token_distribution():
    """
    API endpoint to compute the distribution of the next possible tokens for the given
    input text using the specified model. It returns the top 50 tokens and their
    probabilities along with the most likely next token.

    Query Parameters:
        model_name (str): The name of the model to use for prediction.
        text (str): The input text to predict the next token for.

    Returns:
        JSON: A JSON object containing the next token, top tokens, and their probabilities.
    """
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

    top_k = 50
    top_probs, top_indices = torch.topk(next_token_distribution, k=top_k)

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
    """
    API endpoint to extract and return the attention matrices from the model for the
    given input text.

    Query Parameters:
        model_name (str): The name of the model to use for generating attention filters.
        text (str): The input text to analyze.

    Returns:
        JSON: A JSON object containing the attention matrices.
    """
    model_name = request.args.get("model_name")
    input_text = request.args.get("text")

    if not model_name or not input_text:
        return jsonify({"error": "Missing model_name or text parameter"}), 400

    model, tokenizer = load_model_and_tokenizer(model_name)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    attention_matrices = outputs.attentions

    attention_matrices_list = [att.cpu().numpy().tolist() for att in attention_matrices]

    return jsonify({"attention_matrices": attention_matrices_list})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

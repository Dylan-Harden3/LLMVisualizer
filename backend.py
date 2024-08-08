import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_model_and_tokenizer(model_name):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_API_TOKEN)

    # Load the model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        output_attentions=True,
        use_auth_token=HF_API_TOKEN,
    )

    return model, tokenizer


def get_next_token_distribution(model_name, input_text):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits for the next token distribution
    next_token_logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)
    next_token_distribution = torch.nn.functional.softmax(next_token_logits, dim=-1)

    # Get the predicted next token
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode(next_token_id)

    return next_token, next_token_distribution


def get_attention_filters(model_name, input_text):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model(**inputs)

    # The 'attentions' output is a tuple of attention tensors for each layer
    attention_matrices = outputs.attentions

    return attention_matrices


# get_attention_filters("meta-llama/Meta-Llama-3.1-8B-Instruct", "Hi, my name is")
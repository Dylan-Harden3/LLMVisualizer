import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv

load_dotenv()

def load_model_and_tokenizer(model_name):
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)

    # Load the model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True,
        token=HF_API_TOKEN
    )

    return model, tokenizer

def run_inference(model, tokenizer, input_text):
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
    # Get attention matrices
    # The 'attentions' output is a tuple of attention tensors for each layer
    attention_matrices = outputs.attentions

    return next_token, next_token_distribution, attention_matrices

# Example usage
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # You can change this to any model you prefer
input_text = "The quick brown fox"

model, tokenizer = load_model_and_tokenizer(model_name)
generated_text, next_token_dist, attention_matrices = run_inference(model, tokenizer, input_text)

print(f"Generated text: {generated_text}")
print(f"Next token distribution shape: {next_token_dist.shape}")
print(f"Number of attention layers: {len(attention_matrices)}")
print(f"Shape of first attention matrix: {attention_matrices[0].shape}")
import numpy as np
from collections import Counter

def get_mock_distribution(num_tokens=100):
    """Generate a mock token distribution."""
    probabilities = np.random.dirichlet(np.ones(num_tokens), size=1)[0]
    tokens = [f"token_{i}" for i in range(num_tokens)]
    return dict(zip(tokens, probabilities))

def get_context_based_distribution(text, num_tokens=100):
    """Generate a mock distribution based on the input text."""
    # Simple frequency-based distribution
    words = text.lower().split()
    word_freq = Counter(words)
    total = sum(word_freq.values())
    
    # Create a distribution based on word frequencies
    dist = {word: count/total for word, count in word_freq.items()}
    
    # Pad with random tokens if needed
    while len(dist) < num_tokens:
        new_token = f"token_{len(dist)}"
        dist[new_token] = np.random.uniform(0, min(dist.values()))
    
    # Normalize
    total = sum(dist.values())
    return {k: v/total for k, v in dist.items()}

def get_model_distribution(model, text, num_tokens=100):
    """
    Simulate different model behaviors.
    In a real scenario, this would call actual model APIs.
    """
    if model == "GPT-2":
        # Simulate GPT-2 with slightly more uniform distribution
        dist = get_context_based_distribution(text, num_tokens)
        return {k: v**0.8 for k, v in dist.items()}
    elif model == "GPT-3":
        # Simulate GPT-3 with more peaked distribution
        dist = get_context_based_distribution(text, num_tokens)
        return {k: v**1.2 for k, v in dist.items()}
    elif model == "BERT":
        # Simulate BERT with bidirectional context
        words = text.lower().split()
        dist = Counter(words + words[::-1])
        total = sum(dist.values())
        return {k: v/total for k, v in dist.items()}
    else:
        # Default to context-based distribution
        return get_context_based_distribution(text, num_tokens)

def apply_temperature(probs, temperature):
    """Apply temperature to the distribution."""
    tokens = list(probs.keys())
    probs_array = np.array(list(probs.values()))
    probs_array = np.power(probs_array, 1/temperature)
    probs_array /= np.sum(probs_array)
    return dict(zip(tokens, probs_array))

def apply_top_k(probs, k):
    """Apply top-k filtering to the distribution."""
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_probs[:k]
    total = sum(prob for _, prob in top_k)
    return {token: prob/total for token, prob in top_k}

def apply_top_p(probs, p):
    """Apply top-p (nucleus) sampling to the distribution."""
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    cumulative_probs = np.cumsum([prob for _, prob in sorted_probs])
    cutoff = next(i for i, cum_prob in enumerate(cumulative_probs) if cum_prob > p)
    top_p = sorted_probs[:cutoff+1]
    total = sum(prob for _, prob in top_p)
    return {token: prob/total for token, prob in top_p}

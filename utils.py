import torch
import torch.nn.functional as F

def temperature_sampling(logits, temperature):
    """
    Apply temperature scaling to the logits.
    
    Args:
        logits (torch.Tensor): The output logits of your language model.
        temperature (float): The temperature value to apply.
    
    Returns:
        torch.Tensor: The scaled logits.
    """
    scaled_logits = logits / temperature
    return scaled_logits

def top_p_sampling(logits, top_p=0.9):
    """
    Apply top-p (nucleus) sampling to the logits.
    
    Args:
        logits (torch.Tensor): The output logits of your language model.
        top_p (float): The cumulative probability threshold to use for top-p sampling.
    
    Returns:
        torch.Tensor: The logits with top-p sampling applied.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits
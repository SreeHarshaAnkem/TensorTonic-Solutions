import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    Q,K,V: bs, seq_len, d_model
    Q@K.T: bs, seq_len, seq_len
    
    """
    d_k = Q.size(-1)
    alpha = torch.softmax(Q @ K.transpose(-2,-1)/(d_k**(1/2)), dim=-1)
    return alpha @ V
    
    
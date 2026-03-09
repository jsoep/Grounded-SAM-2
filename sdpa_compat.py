"""
Compatibility shim for PyTorch < 2.0
=====================================
Monkey-patches `torch.nn.functional.scaled_dot_product_attention` and
`torch.backends.cuda.sdp_kernel` so that SAM2 code works on PyTorch 1.12+.

Import this module BEFORE importing sam2:
    import sdpa_compat  # noqa: F401  (patches torch in-place)
    from sam2.build_sam import build_sam2
"""

import torch
import torch.nn.functional as F
import math
import contextlib


def _scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Manual implementation of scaled dot-product attention.
    Equivalent to F.scaled_dot_product_attention in PyTorch 2.0+.
    
    Args:
        query:  (B, num_heads, L, E)
        key:    (B, num_heads, S, E)
        value:  (B, num_heads, S, Ev)
    Returns:
        output: (B, num_heads, L, Ev)
    """
    E = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    
    # (B, num_heads, L, S)
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if is_causal:
        L, S = query.shape[-2], key.shape[-2]
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1
        )
        attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))
    
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    
    return torch.matmul(attn_weight, value)


# Patch F.scaled_dot_product_attention if it doesn't exist
if not hasattr(F, 'scaled_dot_product_attention'):
    F.scaled_dot_product_attention = _scaled_dot_product_attention
    print("[sdpa_compat] Patched torch.nn.functional.scaled_dot_product_attention (PyTorch < 2.0)")

# Patch torch.backends.cuda.sdp_kernel if it doesn't exist
if not hasattr(torch.backends.cuda, 'sdp_kernel'):
    @contextlib.contextmanager
    def _sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        """No-op context manager for PyTorch < 2.0."""
        yield
    
    torch.backends.cuda.sdp_kernel = _sdp_kernel
    print("[sdpa_compat] Patched torch.backends.cuda.sdp_kernel (PyTorch < 2.0)")

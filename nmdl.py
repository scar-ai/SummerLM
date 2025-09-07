import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Utilities ------------------------------------

def exists(x):
    return x is not None

# ----------------------------- Normalization --------------------------------

class RMSNorm(nn.Module):
    """RMS LayerNorm (no centering)"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dim = x.shape[-1]
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        return x * self.scale

# ----------------------------- SwiGLU MLP ----------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim, expansion_factor=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.dropout(self.fc2(x))
        return x

# ----------------------------- Rotary PosEmb --------------------------------

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.cache_seq_len = 0
        self.cache = None

    def forward(self, seq_len, device):
        if self.cache is None or seq_len > self.cache_seq_len or self.cache.device != device:
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cache = emb.unsqueeze(0)
            self.cache_seq_len = seq_len
        return self.cache

@torch.jit.script
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary(x, rot):
    return (x * rot.cos()) + (rotate_half(x) * rot.sin())

# ----------------------------- ALiBi bias ----------------------------------

def build_alibi_slopes(n_heads: int):
    def get_slopes(n):
        def pow2floor(x):
            return 1 << (x.bit_length() - 1)
        m = pow2floor(n)
        slopes = [2 ** (-(2 ** -(math.log2(m) - 3)) * i) for i in range(1, m + 1)]
        if m != n:
            extra = build_alibi_slopes(2 * m)[0::2][: n - m]
            slopes.extend(extra)
        return torch.tensor(slopes)
    return get_slopes(n_heads)

# ----------------------------- Attention -----------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=8,
        head_dim=None,
        dropout=0.0,
        use_rotary=True,
        use_alibi=True
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim or (dim // n_heads)
        assert (self.head_dim * n_heads) == dim, 'dim must be divisible by n_heads when head_dim not provided'

        self.scale = 1.0 / math.sqrt(self.head_dim)

        
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_rotary = use_rotary
        self.use_alibi = use_alibi
        if use_rotary:
            self.rotary = RotaryPositionalEmbedding(self.head_dim)
        else:
            self.rotary = None

        if use_alibi:
            slopes = build_alibi_slopes(self.n_heads)
            self.register_buffer('alibi_slopes', slopes, persistent=False)
        else:
            self.alibi_slopes = None

        self.register_buffer('causal_mask', torch.zeros(0, dtype=torch.bool), persistent=False)
        self.max_cached = 0

    def _get_causal_mask(self, seq_len, device):
        if seq_len > self.max_cached or self.causal_mask.device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            self.causal_mask = mask
            self.max_cached = seq_len
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape -> (B, n_heads, T, head_dim)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary and self.rotary is not None:
            rot = self.rotary(t, x.device)
            # rot shape: (1, T, head_dim)
            # expand to heads: (1, 1, T, head_dim)
            rot = rot.unsqueeze(1)
            q = apply_rotary(q, rot)
            k = apply_rotary(k, rot)

        try:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None if attn_mask is None else attn_mask, is_causal=True, dropout_p=float(self.dropout.p))
            attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, -1)
            return self.o_proj(attn_output)
        except Exception:
            q = q * self.scale
            attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, T, T)

            if self.use_alibi and exists(self.alibi_slopes):
                positions = torch.arange(t, device=x.device).unsqueeze(0)
                rel = positions.transpose(0, 1) - positions
                distances = torch.arange(t, device=x.device).unsqueeze(0) - torch.arange(t, device=x.device).unsqueeze(1)
                slopes = self.alibi_slopes.to(x.device).unsqueeze(1).unsqueeze(2)  # (heads,1,1)
                bias = slopes * distances.unsqueeze(0).float()
                attn_scores = attn_scores + bias

            causal = self._get_causal_mask(t, x.device)
            attn_scores = attn_scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, -1)
            return self.o_proj(attn_output)

# ----------------------------- Transformer Block ---------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=8,
        mlp_expansion=4.0,
        dropout=0.0,
        use_rotary=True,
        use_alibi=True,
        layer_scale_init=1e-2,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads=n_heads, dropout=dropout, use_rotary=use_rotary, use_alibi=use_alibi)
        self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, expansion_factor=mlp_expansion, dropout=dropout)
        self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        if exists(self.layer_scale_1):
            x = x * self.layer_scale_1
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if exists(self.layer_scale_2):
            x = x * self.layer_scale_2
        x = x + residual
        return x

# ----------------------------- Full Model ----------------------------------

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=1024,
        n_heads=16,
        n_layers=24,
        mlp_expansion=4.0,
        dropout=0.1,
        max_seq_len=2048,
        use_rotary=True,
        use_alibi=True,
        ignore_index=-100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads=n_heads, mlp_expansion=mlp_expansion, dropout=dropout, use_rotary=use_rotary, use_alibi=use_alibi)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        # idx: (B, T)
        b, t = idx.shape
        x = self.token_emb(idx)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1), ignore_index=self.ignore_index)
        return logits, loss


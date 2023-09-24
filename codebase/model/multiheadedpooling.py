from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/tomhosking/torchseq/blob/main/torchseq/models/pooling.py
class MultiHeadedPooling(nn.Module):
    def __init__(
        self,
        num_heads: int,
        model_dim: int,
        dropout: Optional[float] = 0.1,
        model_dim_out: Optional[int] = None,
        use_final_linear: Optional[bool] = True,
        use_bilinear: Optional[bool] = False,
        use_layer_norm: Optional[bool] = False
    ) -> None:
        super(MultiHeadedPooling, self).__init__()

        # Dimensionality divisible by number of attention ehads
        assert model_dim % num_heads == 0, f"Number of heads ({num_heads}) does not divide cleanly the given model_dim ({model_dim})!"

        # Size of each individual head
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        # Encoding dimension
        self.model_dim = model_dim
        self.model_dim_out = model_dim_out if model_dim_out is not None else model_dim # Optional
        
        # Actual Pooling Weights
        self.linear_keys = nn.Linear(model_dim, num_heads)
        self.bilinear_keys = nn.Bilinear(model_dim, model_dim, num_heads) if use_bilinear else None
        self.linear_values = nn.Linear(model_dim, num_heads * self.dim_per_head)

        # Model dropout
        self.dropout_rate = dropout

        self.use_final_linear = use_final_linear
        if use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim_out)
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(model_dim)

    def forward(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        query: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        def shape(x, dim=dim_per_head):
            """projection"""
            return x.view(batch_size, -1, num_heads, dim).transpose(1, 2)

        def unshape(x, dim=dim_per_head):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * dim)

        scores = self.linear_keys(key) if query is None else self.bilinear_keys(key, query)
        value = self.linear_values(value)

        scores = shape(scores, 1).squeeze(-1)
        value = shape(value)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(~mask, float("-1e9"))

        # 3) Apply attention dropout and compute context vectors.
        attn = F.softmax(scores, dim=-1)
        drop_attn = F.dropout(attn, p=self.dropout_rate)
        context = torch.sum((drop_attn.unsqueeze(-1) * value), -2)
        context = unshape(context).squeeze(1)

        if self.use_final_linear:
            context = self.final_linear(context)
        if self.use_layer_norm:
            context = self.final_layer_norm(context)

        # shape: (batch_size, 1, model_dim)
        return context.unsqueeze(1)  

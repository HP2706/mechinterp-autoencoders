from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from pydantic import BaseModel, field_validator
from autoencoder import AutoEncoder
class EmbedConfig(BaseModel):
    d_model: int
    d_vocab: int
    init_range: float
    debug: bool 

class PosEmbedConfig(BaseModel):
    d_model: int
    n_ctx: int
    init_range: float
    debug: bool 

class TrainConfig(BaseModel):
    n_epochs: int
    batch_size: int
    lr: float
    beta1: float
    beta2: float
    device: str
    n_tokens : Optional[int] = None

class TransformerConfig(TrainConfig, EmbedConfig, PosEmbedConfig):
    d_model: int
    debug: bool 
    layer_norm_eps: float
    n_ctx: int 
    d_mlp: int
    n_layers: int
    d_head: int 
    n_heads: int
    d_type : str
    tokenizer_name : str


    @field_validator('d_type')
    def check_d_type(cls, value):
        if value not in ['float32', 'float16']:
            raise ValueError('d_type must be either float32 or float16')
        return value

    def get_dtype(self):
        return torch.float32 if self.d_type == 'float32' else torch.float16

class Embed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Embedding(cfg.d_vocab, cfg.d_model, dtype=cfg.get_dtype())
        nn.init.normal_(self.W_E.weight, std=self.cfg.init_range)

    def forward(self, tokens : torch.Tensor):
        if self.cfg.debug:
            print("Tokens:", tokens.shape)
            print("Max token index:", tokens.max().item())
            print("max token in weights:", self.W_E.weight.max().item())
        embed = self.W_E(tokens)  # [batch, position, d_model]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed
    
class UnEmbed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        # Create an embedding layer for the unembedding process
        self.W_D = nn.Embedding(cfg.d_vocab, cfg.d_model, dtype=cfg.get_dtype())
        # Initialize the weights of the embedding layer
        nn.init.normal_(self.W_D.weight, std=self.cfg.init_range)

    def forward(self, logits: torch.Tensor):
        # logits: [batch, position, d_vocab]
        if self.cfg.debug: print("Logits:", logits.shape)
        # Use embedding layer's weight transposed for the matrix multiplication
        unembed = torch.matmul(logits, self.W_D.weight.transpose(0, 1))
        if self.cfg.debug: print("Unembeddings:", unembed.shape)
        return unembed

class PosEmbed(nn.Module):
    def __init__(self, cfg : TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model), dtype=cfg.get_dtype()))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: 
            print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: 
            print("pos_embed:", pos_embed.shape)
        return pos_embed
    

class LayerNorm(nn.Module):
    def __init__(self, cfg : TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + self.cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized

class Attention(nn.Module):
    def __init__(self, cfg : TransformerConfig):
        super(Attention, self).__init__()
        self.cfg = cfg
        # Using nn.Linear for Query, Key, Value, and Output projections
        self.proj_Q = nn.Linear(cfg.d_model, cfg.d_head * cfg.n_heads, dtype=cfg.get_dtype())
        self.proj_K = nn.Linear(cfg.d_model, cfg.d_head * cfg.n_heads, dtype=cfg.get_dtype())
        self.proj_V = nn.Linear(cfg.d_model, cfg.d_head * cfg.n_heads, dtype=cfg.get_dtype())
        self.proj_O = nn.Linear(cfg.d_head * cfg.n_heads, cfg.d_model, dtype=cfg.get_dtype())

        #proper init
        for module in [self.proj_Q, self.proj_K, self.proj_V, self.proj_O]:
            nn.init.uniform_(module.weight, -cfg.init_range, cfg.init_range)

    def forward(self, normalized_resid_pre):
        batch_size, seq_len, _ = normalized_resid_pre.size()

        if self.cfg.debug: print("normalized_resid_pre:", normalized_resid_pre.shape, "dtype:", normalized_resid_pre.dtype)
        if self.cfg.debug: print("Q weights:", self.proj_Q.weight.shape, "dtype:", self.proj_Q.weight.dtype)
        # Compute Q, K, V for all heads
        Q = self.proj_Q(normalized_resid_pre)
        K = self.proj_K(normalized_resid_pre)
        V = self.proj_V(normalized_resid_pre)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_output = torch.bmm(attn_weights, V)
        output = self.proj_O(attn_output)

        if self.cfg.debug: print("attn_out:", output.shape)
        return output
    
    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
        return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg : TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp, dtype=cfg.get_dtype())
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model, dtype=cfg.get_dtype())
        print("mlp max, min W_in", self.W_in.weight.max(), self.W_in.weight.min())
        print("mlp max, min W_out", self.W_out.weight.max(), self.W_out.weight.min())

        #init
        for module in [self.W_in, self.W_out]:
            nn.init.uniform_(module.weight, -cfg.init_range, cfg.init_range)

    def forward(self, normalized_resid_mid, return_hidden : bool) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: 
            print("Normalized_resid_mid:", normalized_resid_mid.shape)
        mlp_inner = self.W_in(normalized_resid_mid)
        mlp_out = F.gelu(mlp_inner)
        mlp_out = self.W_out(mlp_out)
        if return_hidden:
            return mlp_out, mlp_inner
        else:
            return mlp_out, None

class TransformerBlock(nn.Module):
    def __init__(self, cfg : TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, residual, return_hidden : bool)-> tuple[torch.Tensor, Optional[torch.Tensor]]:
        normalized_resid_pre = self.ln1(residual)
        print()
        attn_out = self.attn(normalized_resid_pre)
        normalized_resid_mid = normalized_resid_pre + attn_out
        normalized_resid_post = self.ln2(normalized_resid_mid)
        mlp_out, mlp_inner = self.mlp(normalized_resid_post, return_hidden)
        residual = normalized_resid_mid + mlp_out
        return (residual, mlp_inner)

class Transformer(nn.Module):
    def __init__(self, cfg : TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.unembed = UnEmbed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_tokens: int) -> torch.Tensor:
        i = 0
        while i < n_tokens:
            logits, _ = self.forward(tokens, return_hidden=False)
            # Clipping logits to a reasonable range before softmax to prevent underflow
            logits_clipped = torch.clamp(logits, min=-10, max=10)
            distb = logits_clipped.log_softmax(dim=-1)
            next_token = torch.multinomial(distb[:, -1].exp(), 1)  # Using exp to convert log probabilities to probabilities
            tokens = torch.cat((tokens, next_token), dim=-1)  # Concatenate [batch, seq_len + 1]
            i += 1
        return tokens

    def forward(self, tokens, return_hidden : bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        #[batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            (residual, mlp_hidden_acts) = block(residual, return_hidden)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        return (logits, mlp_hidden_acts) # type: ignore


    def forward_feature_ablate(self, tokens, auto_encoder : AutoEncoder, ablate_feature : int):
        #[batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            (residual, mlp_hidden_acts) = block(residual, return_hidden = True)
            reconstructed_x = auto_encoder.forward(mlp_hidden_acts, method = 'reconstructed')
            residual[:, :, ablate_feature] = reconstructed_x[:, :, ablate_feature] #type: ignore
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        #compute p(s)
        p_s = torch.prod(logits, dim=-1)
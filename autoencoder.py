from typing import Optional
from statistics import geometric_mean
import torch
from torch import Tensor
import torch.nn as nn
from pydantic import BaseModel, field_validator
from torch.nn import functional as F
from typing import Tuple, Union

class AutoencoderConfig(BaseModel):
    seed: int
    batch_size: int
    buffer_mult: int
    lr: float
    num_tokens: Optional[int] = None
    l1_coeff: float
    beta1: float
    beta2: float
    dict_mult: int
    seq_len: int
    d_mlp: int
    enc_dtype: str = 'float32'
    remove_rare_dir: bool
    batch_size: int
    buffer_size: int
    buffer_batches: int
    device : str 

    @field_validator('enc_dtype')
    def check_d_type(cls, value):
        if value not in ['float32', 'float16']:
            raise ValueError('d_type must be either float32 or float16')
        return value
    
    @field_validator('device')
    def check_device(cls, value):
        if value not in ['cuda', 'mps']:
            raise ValueError(f'device must be either cuda or mps got {value}')
        return value
    

    def get_dtype(self):
        return torch.float32 if self.enc_dtype == 'float32' else torch.float16

#inspired by neel nanda https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=qCF9odNdAvKX
class AutoEncoder(nn.Module):
    def __init__(self, cfg : AutoencoderConfig, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg.d_mlp * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, d_hidden, dtype=cfg.get_dtype())))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_mlp, dtype=cfg.get_dtype())))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.get_dtype())) # initialize to zero
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.get_dtype())) # initialize to zero
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = cfg.l1_coeff

    def forward(self, x, method : str = 'with_loss')-> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        if method == 'with_loss':
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            l1_loss = self.l1_coeff * (acts.float().abs().sum())
            loss = l2_loss + l1_loss
            return loss, x_reconstruct, acts, l2_loss, l1_loss
        else:
            return x_reconstruct  

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

#from geom_median.torch import compute_geometric_median 
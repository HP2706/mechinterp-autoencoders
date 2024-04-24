from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from pydantic import BaseModel, field_validator
from torch.nn import functional as F
from typing import Tuple, Union
from mechninterp_utils import utils
import pprint
#internal imports
from mechninterp_utils import replacement_hook, zero_ablate_hook, mean_ablate_hook


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
    enc_dtype: str = 'fp32'
    batch_size: int
    buffer_size: int
    buffer_batches: int
    device : str 
    n_epochs: Optional[int] = None


    @field_validator('enc_dtype')
    def check_d_type(cls, value):
        if value not in ['fp32', 'fp16']:
            raise ValueError('d_type must be either fp32 or fp16')
        return value
    
    @field_validator('device')
    def check_device(cls, value):
        if value not in ['cuda', 'mps']:
            raise ValueError(f'device must be either cuda or mps got {value}')
        return value

    def get_dtype(self):
        return torch.float32 if self.enc_dtype == 'fp32' else torch.float16

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
        if method == 'with_acts':
            return acts
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

    @classmethod
    def load_from_hf(cls, version, device : str):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47

        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json") # type: ignore
        cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
        cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
        cfg["device"] = device
        valid_cfg = AutoencoderConfig(**cfg)
        self = cls(cfg=valid_cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True)) # type: ignore
        return self

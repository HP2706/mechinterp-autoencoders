from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from pydantic import BaseModel, field_validator
from torch.nn import functional as F
from typing import Tuple, Union, Literal
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
    
    @property 
    def dtype(self):
        return torch.float32 if self.enc_dtype == 'fp32' else torch.float16

#inspired by neel nanda https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=qCF9odNdAvKX
class AutoEncoder(nn.Module):
    def __init__(self, cfg : AutoencoderConfig, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg.d_mlp * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) # initialize to zero
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype)) # initialize to zero
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = cfg.l1_coeff

    def forward(
        self,
        x, 
        method : Literal[
            'with_acts', 'with_loss', 'reconstruct', 'with_new_loss'
        ] = 'with_loss'
    )-> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        
        x_center = x - self.b_dec
        acts = F.relu(x_center @ self.W_enc + self.b_enc)
        if method == 'with_acts':
            return acts
        x_reconstruct = acts @ self.W_dec + self.b_dec
        if method == 'with_loss':
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            l1_loss = self.l1_coeff * (acts.float().abs().sum())
            loss = l2_loss + l1_loss
            return loss, x_reconstruct, acts, l2_loss, l1_loss
        
        #new anthropic technique for improved autoencoder training
        elif method == 'with_new_loss':
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            l1_loss = torch.sum(
                torch.abs(acts) 
                * 
                torch.norm(self.W_dec, dim=0, p=2)
            )
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

#from paper https://arxiv.org/pdf/2404.16014
class GatedAutoEncoder(nn.Module):
    def __init__(self, cfg : AutoencoderConfig, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg.d_mlp * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.relu = nn.ReLU()
        # Initialize parameters for the gated autoencoder
        self.l1_coeff = cfg.l1_coeff
        self.W_gate = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, d_hidden, dtype=cfg.dtype)))

        self.b_gate = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))
        self.b_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))

        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype)) # initialize to zero

        # Initialize W_mag as a vector of learnable parameters
        self.r_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) #TODO should this be initialized to to zero?
        self.W_mag = torch.exp(self.r_mag)[None, :] * self.W_gate
        self.W_mag = nn.Parameter(self.W_mag)

    #TODO: Implement forward and loss
    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct']
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        # Ensure x is [batch_size, feature_size]
        x_center = x - self.b_dec  # [batch_size, feature_size]
        gate_center = x_center @ self.W_gate + self.b_gate
        active_features = (gate_center > 0 ).float()  # [batch_size, d_hidden]
        feature_magnitudes = self.relu(x_center @ self.W_mag + self.b_mag)  # [batch_size, d_hidden]
        acts = active_features * feature_magnitudes  # [batch_size, d_hidden]
        x_reconstruct = acts @ self.W_dec + self.b_dec  # [batch_size, feature_size]
        
        if method == 'with_acts':
            return acts
        elif method == 'reconstruct':
            return x_reconstruct
        elif method == 'with_loss':
            L_Reconstruct = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            via_gate_feature_magnitudes = self.relu(gate_center)  # [batch_size, d_hidden]
            L_Sparsity = self.l1_coeff * (via_gate_feature_magnitudes.float().sum())
            # Frozen decoder for L_aux
            with torch.no_grad():
                W_dec_frozen = self.W_dec.detach()
                b_dec_frozen = self.b_dec.detach()
            via_gate_reconstruction = (via_gate_feature_magnitudes @ W_dec_frozen + b_dec_frozen)
            L_aux = (x - via_gate_reconstruction).pow(2).sum(-1).mean(0)

            loss = L_Reconstruct + L_Sparsity + L_aux
            return loss, x_reconstruct, acts, L_Reconstruct, L_Sparsity, L_aux
        else:
            return x_reconstruct

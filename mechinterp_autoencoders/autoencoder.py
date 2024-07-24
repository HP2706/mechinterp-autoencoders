from torch.optim import Optimizer
from tqdm import tqdm
from typing import Optional
import torch
from torch import Tensor, nn
from pydantic import BaseModel, model_validator, Field
from torch.nn import functional as F
from typing import Callable, Union, Literal, List, Any, Type, TypeVar
from jaxtyping import Float, Int
from .base_autoencoder import AutoEncoderBase, AutoEncoderBaseConfig
from .compute_metrics import normalized_L1_loss, l1_norm, mean_absolute_error, l0_norm, did_fire, avg_num_firing_per_neuron

T = TypeVar('T', bound='AutoEncoderBase')

class AutoEncoderConfig(AutoEncoderBaseConfig):
    l1_coeff : float

class AutoEncoder(AutoEncoderBase):
    def __init__(
        self, 
        cfg : AutoEncoderConfig
    ):
        super().__init__(cfg)
        self.d_hidden = cfg.d_input * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.W_enc = torch.randn(self.cfg.d_input, self.d_hidden, dtype=self.cfg.enc_dtype)
        self.pre_bias = torch.randn(self.cfg.d_input, dtype=self.cfg.enc_dtype)
        self.W_dec = torch.randn(self.d_hidden, self.cfg.d_input, dtype=self.cfg.enc_dtype)
        self.b_enc = torch.randn(self.d_hidden, dtype=self.cfg.enc_dtype)
        self.activation = nn.ReLU()
        self.l1_coeff = cfg.l1_coeff
        self.to(self.cfg.device) # move to device
        
    
    def _prepare_params(self, feature_indices: Optional[slice]):
        if feature_indices is None:
            return self.W_enc, self.pre_bias, self.W_dec, self.b_enc
        
        W_enc = self.W_enc[feature_indices, :]
        pre_bias = self.pre_bias[feature_indices]
        W_dec = self.W_dec[:, feature_indices]
        b_enc = self.b_enc
        return W_enc, pre_bias, W_dec, b_enc

    def encode(self, x: Tensor, feature_indices: Optional[slice] = None) -> Tensor:
        W_enc, pre_bias, W_dec, b_enc = self._prepare_params(feature_indices)
        x_center = x - pre_bias
        acts = self.activation(x_center @ W_enc + b_enc)
        return acts

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, dict]:
        W_enc, pre_bias, W_dec, b_enc = self._prepare_params(feature_indices)
        acts = self.encode(x, feature_indices)
        if method == 'with_acts':
            return acts

        x_reconstruct = acts @ W_dec + pre_bias
        if method in ['with_loss', 'with_new_loss']:
            l2_loss = normalized_L1_loss(x_reconstruct, x)
            if method == 'with_loss':
                L_sparse = l1_norm(acts)
            else:
                L_sparse = (acts.abs() * torch.norm(W_dec, dim=-1)).sum()
            
            loss = l2_loss + (self.l1_coeff * L_sparse)
            
            return {
                "loss": loss, 
                "x_reconstruct": x_reconstruct, 
                "acts": acts, 
                "acts_sum": acts.sum(1).mean(),
                "l2_loss": l2_loss, 
                "l_sparsity": L_sparse,
                "l1_loss": mean_absolute_error(x_reconstruct, x),
                "normalized_l1_loss": normalized_L1_loss(acts, x),
                "l0_norm": l0_norm(acts),
                "did_fire": did_fire(acts),
                "avg_num_firing_per_neuron": avg_num_firing_per_neuron(acts)
            }
        elif method == 'reconstruct':
            return x_reconstruct
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def decode(self, acts : Tensor) -> Tensor:
        return acts @ self.W_dec + self.pre_bias
    
    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):
        for dict_idx, (k, v) in tqdm(enumerate(optimizer.state.items()), desc="setting gradients to zero"):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_sae, self.cfg.d_input), f"expected shape (self.d_sae, self.cfg.d_input) got {k.data.shape}"
                    v[v_key][indices, :] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.cfg.d_input, self.d_sae), f"expected shape (self.cfg.d_input,) got {k.data.shape}"
                    v[v_key][:, indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae,), f"expected shape (self.cfg.d_input, self.d_sae) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.cfg.d_input,), f"expected shape (self.d_sae,) got {k.data.shape}"
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
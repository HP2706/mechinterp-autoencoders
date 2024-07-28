from beartype import beartype
from typing import Optional
import torch
from torch import Tensor, nn
from typing import  Union, Literal, TypeVar
from jaxtyping import Float, jaxtyped
from .base_autoencoder import BaseAutoEncoder, AutoEncoderBaseConfig, AbstractAutoEncoder
from .compute_metrics import normalized_L1_loss, l1_norm, mean_absolute_error, l0_norm, did_fire, avg_num_firing_per_neuron

class AutoEncoderConfig(AutoEncoderBaseConfig):
    l1_coeff : float

class AutoEncoder(BaseAutoEncoder):
    def __init__(
        self, 
        cfg : AutoEncoderConfig
    ):
        super().__init__()
        self.cfg = cfg
        self.d_hidden = cfg.d_input * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.initialize_weights()
        self.activation = nn.ReLU()
        self.l1_coeff = cfg.l1_coeff
        self.to(self.cfg.device) # move to device

    def encode(
        self, 
        x: Float[Tensor, "batch_size d_input"], 
        feature_indices: Optional[slice] = None
    ) -> Float[Tensor, "batch_size d_hidden"]:
        with self._prepare_params(x, feature_indices):
            x_center = x - self.pre_bias
            acts = self.activation(x_center @ self.W_enc + self.b_enc)
        return acts

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, dict]:
        with self._prepare_params(x, feature_indices):
            acts = self.encode(x, feature_indices)
            if method == 'with_acts':
                return acts

            x_reconstruct = self.decode(acts, feature_indices)
            if method in ['with_loss', 'with_new_loss']:
                l2_loss = normalized_L1_loss(x_reconstruct, x)
                if method == 'with_loss':
                    L_sparse = l1_norm(acts)
                else:
                    L_sparse = (acts.abs() * torch.norm(self.W_dec, dim=-1)).sum()
                
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
        

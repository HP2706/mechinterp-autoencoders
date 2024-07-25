import torch
from contextlib import contextmanager
from torch import nn, Tensor
from .base_autoencoder import AutoEncoderBase, AutoEncoderBaseConfig
from jaxtyping import jaxtyped, Float
from beartype import beartype
from typing import Optional, Union, Literal
from .compute_metrics import normalized_mse, mean_absolute_error, normalized_L1_loss, l0_norm, did_fire, avg_num_firing_per_neuron
from tqdm import tqdm
from torch.optim import Optimizer

class GatedAutoEncoderConfig(AutoEncoderBaseConfig):
    l1_coeff: float

#from paper https://arxiv.org/pdf/2404.16014
class GatedAutoEncoder(AutoEncoderBase):
    def __init__(self, cfg : GatedAutoEncoderConfig):
        super().__init__(cfg)
        d_hidden = cfg.d_input * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        # Initialize parameters for the gated autoencoder
        self.l1_coeff = cfg.l1_coeff
        self.W_gate = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_input, d_hidden, dtype=cfg.dtype)))

        self.b_gate = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))
        self.b_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))

        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_input, dtype=cfg.dtype)))
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_input, dtype=cfg.dtype)) # initialize to zero

        # Initialize W_mag as a vector of learnable parameters
        self.r_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) #TODO should this be initialized to zero?
        weight_init = torch.exp(self.r_mag)[None, :] * self.W_gate
        self.W_mag = nn.Parameter(weight_init)
        self.activation = nn.ReLU()
        self.to(self.cfg.device)

    @contextmanager
    def _prepare_params(self, feature_indices: Optional[slice] = None):
        if feature_indices is None:
            yield
        else:
            #we store a copy
            #this might not be great for memory consumption though
            original_W_mag = self.W_mag
            original_W_gate = self.W_gate
            original_pre_bias = self.pre_bias
            original_W_dec = self.W_dec
            original_b_gate = self.b_gate
            original_b_mag = self.b_mag
            
            self.W_mag = self.W_mag[feature_indices, :]
            self.W_gate = self.W_gate[feature_indices, :]
            self.pre_bias = self.pre_bias[feature_indices]
            self.W_dec = self.W_dec[:, feature_indices]
            self.b_gate = self.b_gate[feature_indices]
            self.b_mag = self.b_mag[feature_indices]
            
            try:
                yield
            finally:
                self.W_mag = original_W_mag
                self.W_gate = original_W_gate
                self.pre_bias = original_pre_bias
                self.W_dec = original_W_dec
                self.b_gate = original_b_gate
                self.b_mag = original_b_mag
    
    @jaxtyped(typechecker=beartype)
    def encode(
        self, 
        x: Float[Tensor, "batch d_input"], 
        feature_indices: Optional[slice] = None
    ) -> tuple[
            Float[Tensor, "batch d_hidden"], 
            Float[Tensor, "batch d_hidden"]
        ]:
        with self._prepare_params(feature_indices):
            x = x[:, feature_indices] if feature_indices is not None else x

            x_center = x - self.pre_bias
            gate_center = x_center @ self.W_gate + self.b_gate
            active_features = (gate_center > 0).to(self.cfg.dtype)
            feature_magnitudes = self.activation(x_center @ self.W_mag + self.b_mag)
            # Computing final activations
            acts = active_features * feature_magnitudes
        return acts, gate_center
    
    @jaxtyped(typechecker=beartype)
    def decode(
        self, 
        acts: Float[Tensor, "batch d_hidden"], 
        feature_indices: Optional[slice] = None
    ) -> Float[Tensor, "batch d_input"]:
        with self._prepare_params(feature_indices):
            return acts @ self.W_dec + self.pre_bias

    @jaxtyped(typechecker=beartype)
    def forward(
        self, 
        x: Float[Tensor, "batch d_input"], 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, dict]:
        acts, gate_center = self.encode(x, feature_indices)

        if method == 'with_acts':
            return acts
        
        x_reconstruct = self.decode(acts, feature_indices)
        
        if method == 'reconstruct':
            return x_reconstruct
        elif method == 'with_loss':
            # Reconstruct loss
            l2 = normalized_mse(x_reconstruct, x)
            # Computing Sparsity loss
            via_gate_feature_magnitudes = self.activation(gate_center)
            L_Sparsity = via_gate_feature_magnitudes.float().sum()

            # Computing L_aux with frozen decoder
            with torch.no_grad():
                W_dec_frozen = self.W_dec.detach()
                b_enc_frozen = self.b_mag.detach()
            via_gate_reconstruction = (via_gate_feature_magnitudes @ W_dec_frozen + b_enc_frozen)
            L_aux = normalized_mse(via_gate_reconstruction, x)
            loss = l2 + (self.l1_coeff * L_Sparsity) + L_aux

            return {
                'loss': loss, 
                'l2_loss': l2,
                'l1_loss': mean_absolute_error(x_reconstruct, x),
                'normalized_l1_loss': normalized_L1_loss(acts, x),
                'l_sparsity': L_Sparsity,
                'l0_norm': l0_norm(acts),
                'x_reconstruct': x_reconstruct, 
                'acts': acts, 
                'acts_sum': acts.sum(),
                'L_aux': L_aux,
                'did_fire': did_fire(acts),
                'avg_num_firing_per_neuron': avg_num_firing_per_neuron(acts),
            }
        else:
            raise ValueError(f"Invalid method: {method}")

    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):
        for dict_idx, (k, v) in tqdm(enumerate(optimizer.state.items()), desc="setting gradients to zero"):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.cfg.d_input, self.d_hidden), f"expected shape ({self.cfg.d_input}, {self.d_hidden}) got {k.data.shape}"
                    v[v_key][:, indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,), f"expected shape (self.d_sae,) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae,), f"expected shape (self.d_sae,) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_sae, self.cfg.d_input), f"expected shape (self.d_sae, self.cfg.d_input) got {k.data.shape}"
                    v[v_key][indices, :] = 0.0
                elif dict_idx == 4:
                    assert k.data.shape == (self.cfg.d_input,), f"expected shape (self.cfg.d_input,) got {k.data.shape}"
                elif dict_idx == 5:
                    assert k.data.shape == (self.cfg.d_input, self.d_sae), f"expected shape (self.cfg.d_input, 1536) got {k.data.shape}"
                    v[v_key][:, indices]
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
        
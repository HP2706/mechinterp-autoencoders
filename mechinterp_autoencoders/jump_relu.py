from contextlib import contextmanager
import math
from typing import Literal, Optional, Union, cast
import torch
from torch import Tensor, nn
from .base_autoencoder import AutoEncoderBaseConfig, AutoEncoderBase
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from .compute_metrics import normalized_mse, mean_absolute_error, normalized_L1_loss, l0_norm, did_fire, avg_num_firing_per_neuron

class JumpReLUAutoEncoderConfig(AutoEncoderBaseConfig):
    l1_coeff: float
    threshold: float = 0.0
    bandwidth: float = 1.0
    use_pre_enc_bias: bool = True
    log_threshold: float = 0.2 #IDK about this param

def rectangle(x : Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

def heaviside(x : Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
    return (x > 0).to(x.dtype)

#inspired by https://storage.googleapis.com/jumprelu-saes-paper/JumpReLU_SAEs.pdf
class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        x : Float[Tensor, "batch d_sae"], 
        threshold : float, 
        bandwidth : float
    ) -> Float[Tensor, "batch d_sae"]:
        ctx.save_for_backward(x, torch.tensor(threshold))
        ctx.bandwidth = bandwidth
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, output_grad):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(x)  # We don't apply STE to x input
        threshold_grad = (
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * output_grad
        )
        return x_grad, threshold_grad, None  # None for bandwidth gradient

class JumpReLUModule(nn.Module):
    def __init__(self, threshold=0.0, bandwidth=1.0):
        super().__init__()
        self.threshold = threshold
        self.bandwidth = bandwidth

    def forward(self, x : Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        return cast(Tensor, JumpReLU.apply(x, self.threshold, self.bandwidth))

class JumpReLUAutoEncoder(AutoEncoderBase):
    def __init__(
        self, 
        cfg : JumpReLUAutoEncoderConfig
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.cfg.d_input, self.d_sae, dtype=self.cfg.dtype)))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.d_sae, self.cfg.d_input, dtype=self.cfg.dtype)))
        if self.cfg.tie_w_dec:
            self.W_enc.data = self.W_dec.data.t().clone()

        self.pre_bias = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=self.cfg.dtype)) # initialize to zero
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype)) # initialize to zero
        self.b_dec = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=self.cfg.dtype)) # initialize to zero
        threshold = math.exp(self.cfg.log_threshold)
        self.jump_relu = JumpReLUModule(threshold, self.cfg.bandwidth)

    @contextmanager
    def _prepare_params(self, x: Optional[Tensor], feature_indices: Optional[slice]):
        if feature_indices is None:
            yield x
        else:
            original_W_enc = self.W_enc
            original_pre_bias = self.pre_bias
            original_W_dec = self.W_dec
            
            self.W_enc = self.W_enc[feature_indices, :]
            self.pre_bias = self.pre_bias[feature_indices]
            self.W_dec = self.W_dec[:, feature_indices]
            
            x = x[:, feature_indices] if x is not None else None
            
            try:
                yield x
            finally:
                self.W_enc = original_W_enc
                self.pre_bias = original_pre_bias
                self.W_dec = original_W_dec

    @jaxtyped(typechecker=beartype)
    def encode_pre_act(
        self,
        x : Float[Tensor, "batch d_input"],
        feature_indices : Optional[slice] = None
    ) -> Float[Tensor, "batch d_sae"]:
        with self._prepare_params(x, feature_indices):
            if self.cfg.use_pre_enc_bias:
                x = x - self.b_dec
            return x @ self.W_enc + self.b_enc
        
    @jaxtyped(typechecker=beartype)
    def encode(
        self,
        x : Float[Tensor, "batch d_input"],
        feature_indices : Optional[slice] = None
    ) -> Float[Tensor, "batch d_sae"]:
        return torch.relu(self.encode_pre_act(x, feature_indices))

    @jaxtyped(typechecker=beartype)
    def decode(
        self,
        x : Float[Tensor, "batch d_sae"],
        feature_indices : Optional[slice] = None
    ) -> Float[Tensor, "batch d_input"]:
        with self._prepare_params(x, feature_indices):
            return x @ self.W_dec + self.b_dec

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x : Float[Tensor, "batch d_input"],
        method : Literal["with_loss", "with_acts", "reconstruct"],
        feature_indices : Optional[slice] = None
    ) -> Union[
            Float[Tensor, "batch d_input"], 
            dict
        ]:
    
        acts = self.encode(x, feature_indices)
        if method == 'with_acts':
            return acts

        x_reconstruct = self.decode(acts)

        if method == 'with_loss':
            reconstruction_error = x - x_reconstruct
            reconstruction_loss = torch.sum(reconstruction_error**2, dim=-1)
            feature_magnitudes = self.jump_relu.forward(acts)

            # Compute per-example sparsity loss
            l0 = heaviside(feature_magnitudes - self.cfg.threshold).sum(-1)
            sparsity_loss = self.cfg.l1_coeff * l0
            # Return the batch-wise mean total loss
            loss = torch.mean(reconstruction_loss + sparsity_loss, dim=0)

            return {
                "loss": loss, 
                "x_reconstruct": x_reconstruct, 
                "sparsity_loss": sparsity_loss,
                "reconstruction_loss": reconstruction_loss,
                "avg_num_firing_per_neuron": avg_num_firing_per_neuron(acts)
            }
            
        elif method == 'reconstruct':
            return x_reconstruct
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def zero_optim_grads(self):
        #TODO
        pass
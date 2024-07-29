import torch
from contextlib import contextmanager
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable, Optional, Union, Literal, overload
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from mechinterp_autoencoders.base_autoencoder import AbstractAutoEncoder, BaseAutoEncoder, AutoEncoderBaseConfig
from mechinterp_autoencoders.compute_metrics import normalized_mse, mean_absolute_error, normalized_L1_loss, l0_norm, did_fire, avg_num_firing_per_neuron

if torch.cuda.is_available():
    from mechinterp_autoencoders.kernels import TritonDecoder

class TopKAutoEncoderConfig(AutoEncoderBaseConfig):
    k: int
    k_aux: int     

#based on https://cdn.openai.com/papers/sparse-autoencoders.pdf
class TopKActivationFn(nn.Module):
    def __init__(
        self, 
        k: int, 
        k_aux : int, 
        postact_fn: Callable = nn.ReLU()
    ) -> None:
        super().__init__()
        self.k = k
        self.k_aux = k_aux
        self.postact_fn = postact_fn

    def forward(
        self, 
        x: Float[Tensor, "batch d_in"]
    ) -> tuple[Float[Tensor, "batch d_in"], Int[Tensor, "batch d_sae"]]:
        #we assume the relu already has been applied
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x, device=x.device)
        result.scatter_(-1, topk.indices, values)
        return result, topk.indices

    def forward_aux(
        self, 
        x: Float[Tensor, "batch d_in"], 
        ema_frequency_counter: Float[Tensor, "d_sae"]
    ) -> tuple[Float[Tensor, "batch d_in"], Int[Tensor, "batch top_k"]]:
        topk = torch.topk(
            ema_frequency_counter, 
            k=self.k_aux, 
            dim=-1, 
            largest=False #NOTE we want the least used features
        )
        dead_features = x[:, topk.indices] 
        values = self.postact_fn(dead_features)
        result = torch.zeros_like(x, device=x.device)
        indices = topk.indices.unsqueeze(0).expand(x.size(0), -1)
        result.scatter_(-1, indices, values)
        return result, indices
   
class TopKAutoEncoder(BaseAutoEncoder):
    cfg: TopKAutoEncoderConfig
    def __init__(
        self, 
        cfg: TopKAutoEncoderConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.initialize_weights()
        self.activation = TopKActivationFn(k=cfg.k, k_aux=cfg.k_aux)
        self.to(self.cfg.dtype)

    def encode_pre_act(
        self,
        x: Float[Tensor, "batch d_in"],
        feature_indices: Optional[slice] = None
    ) -> Float[Tensor, "batch d_sae"]:
        '''encode without activation'''
        with self._prepare_params(x, feature_indices):
            x = x - self.pre_bias
            return x @ self.W_enc + self.b_enc

    def encode(
        self, 
        x: Float[Tensor, "batch d_in"],
        feature_indices: Optional[slice] = None
    ) -> tuple[
            Float[Tensor, "batch d_sae"], 
            Int[Tensor, "..."]
        ]:
        '''encode with activation returns (relu(topk), indices)'''
        with self._prepare_params(x, feature_indices):
            acts = self.encode_pre_act(x)
            return self.activation.forward(acts)

    def decode(
        self, 
        acts: Float[Tensor, "batch d_sae"],
        non_zero_indices: Int[Tensor, "..."],
        feature_indices: Optional[slice] = None
    )-> Float[Tensor, "batch d_in"]:        
        with self._prepare_params(acts, feature_indices):
            top_k_acts = acts.gather(1, non_zero_indices).reshape(non_zero_indices.shape[0], -1)
            if self.cfg.use_kernel and torch.cuda.is_available():
                return self.kernel_decode(
                    non_zero_indices.contiguous(), 
                    top_k_acts
                ) + self.pre_bias
            else:
                #same thing just in pytorch
                return self.eager_decode(non_zero_indices, top_k_acts) + self.pre_bias


    @overload
    def forward(
        self,
        x: Tensor,
        method: Literal['with_acts'],
        ema_frequency_counter: None = None,
        feature_indices: Optional[slice] = None
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        method: Literal['with_loss'],
        ema_frequency_counter: Tensor,
        feature_indices: Optional[slice] = None
    ) -> dict:
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        method: Literal['reconstruct'],
        ema_frequency_counter: None = None,
        feature_indices: Optional[slice] = None
    ) -> Tensor:
        ...

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        ema_frequency_counter : Optional[Tensor] = None,
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, dict]:
        acts, non_zero_indices = self.encode(x, feature_indices)
        if method == 'with_acts':
            return acts

        x_reconstruct = self.decode(acts, non_zero_indices, feature_indices)

        if method == 'with_loss':
            assert ema_frequency_counter is not None, "ema_frequency_counter must be provided for with_loss method"
            assert ema_frequency_counter.device == x.device, f"ema_frequency_counter device {ema_frequency_counter.device} != x device {x.device}"

            l2_loss = normalized_mse(x_reconstruct, x)
            aux_acts, indices = self.activation.forward_aux(acts, ema_frequency_counter)
            e = x - x_reconstruct

            e_hat = self.decode(aux_acts, indices, feature_indices)

            aux_k = normalized_mse(e_hat, e)
            if aux_k.isnan().any():
                aux_k = torch.zeros_like(aux_k)
            alpha = 1/32 #as specified in openai autoencoder paper
            loss = l2_loss + alpha * aux_k
            return {
                "loss": loss, 
                "x_reconstruct": x_reconstruct, 
                "acts": acts, 
                "acts_sum": acts.sum(1).mean(),
                "l2_loss": l2_loss, 
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
import torch
from contextlib import contextmanager
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable, Optional, Union, Literal
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from mechinterp_autoencoders.base_autoencoder import AutoEncoderBase, AutoEncoderBaseConfig
from mechinterp_autoencoders.compute_metrics import normalized_mse, mean_absolute_error, normalized_L1_loss, l0_norm, did_fire, avg_num_firing_per_neuron

if torch.cuda.is_available():
    from kernels import TritonDecoder

class TopKAutoEncoderConfig(AutoEncoderBaseConfig):
    k: int
    k_aux: int     
    use_kernel : bool #use triton kernels

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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        #we assume the relu already has been applied
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x, device=x.device)
        result.scatter_(-1, topk.indices, values)
        return result, topk.indices

    def forward_aux(self, x: Tensor, ema_frequency_counter: Tensor) -> tuple[Tensor, Tensor]:
        topk = torch.topk(
            ema_frequency_counter, 
            k=self.k_aux, 
            dim=-1, 
        )
        dead_features = x[:, topk.indices] 
        values = self.postact_fn(dead_features)
        result = torch.zeros_like(x, device=x.device)
        indices = topk.indices.unsqueeze(0).expand(x.size(0), -1)
        result.scatter_(-1, indices, values)
        return result, indices
   
class TopKAutoEncoder(AutoEncoderBase):
    def __init__(
        self, 
        cfg: TopKAutoEncoderConfig
    ) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.d_hidden = cfg.d_input
        self.d_hidden = cfg.d_input * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.W_enc = nn.Parameter(torch.randn(self.cfg.d_input, self.d_hidden, dtype=self.cfg.dtype))
        self.pre_bias = nn.Parameter(torch.randn(self.cfg.d_input, dtype=self.cfg.dtype))
        self.W_dec = nn.Parameter(torch.randn(self.d_hidden, self.cfg.d_input, dtype=self.cfg.dtype))
        self.b_enc = nn.Parameter(torch.randn(self.d_hidden, dtype=self.cfg.dtype))

        self.activation = TopKActivationFn(k=cfg.k, k_aux=cfg.k_aux)

    @contextmanager
    def _prepare_params(self, feature_indices: Optional[slice]):
        if feature_indices is None:
            yield
        else:
            original_W_enc = self.W_enc
            original_pre_bias = self.pre_bias
            original_W_dec = self.W_dec
            
            self.W_enc = self.W_enc[feature_indices, :]
            self.pre_bias = self.pre_bias[feature_indices]
            self.W_dec = self.W_dec[:, feature_indices]
            
            try:
                yield
            finally:
                self.W_enc = original_W_enc
                self.pre_bias = original_pre_bias
                self.W_dec = original_W_dec

    @jaxtyped(typechecker=beartype)
    def encode_pre_act(
        self,
        x: Float[Tensor, "batch d_in"],
        feature_indices: Optional[slice] = None
    ) -> Float[Tensor, "batch d_sae"]:
        '''encode without activation'''
        with self._prepare_params(feature_indices):
            x = x - self.pre_bias
            return x @ self.W_enc + self.b_enc

    @jaxtyped(typechecker=beartype)
    def encode(
        self, 
        x: Float[Tensor, "batch d_in"],
        feature_indices: Optional[slice] = None
    ) -> tuple[
            Float[Tensor, "batch d_sae"], 
            Int[Tensor, "..."]
        ]:
        '''encode with activation returns (relu(topk), indices)'''
        with self._prepare_params(feature_indices):
            x = x[:, feature_indices] if feature_indices is not None else x
            acts = self.encode_pre_act(x)
            return self.activation.forward(acts)

    @jaxtyped(typechecker=beartype)
    def decode(
        self, 
        acts: Float[Tensor, "batch d_sae"],
        non_zero_indices: Int[Tensor, "..."],
        feature_indices: Optional[slice] = None
    )-> Float[Tensor, "batch d_in"]:
        with self._prepare_params(feature_indices):
            if self.cfg.use_kernel:
                #use custom kernel
                '''
                Args:
                    acts: the activations of the topk
                    non_zero_indices: the indices of the non-zero elements in the activations after topk(this is necessary for sparsity)
                '''
                y = TritonDecoder.apply(non_zero_indices, acts.to(self.cfg.dtype), self.W_dec)
                return y + self.pre_bias
            else:
                return acts @ self.W_dec + self.pre_bias

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        ema_frequency_counter : Tensor,
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, dict]:
        assert ema_frequency_counter.device == x.device, f"ema_frequency_counter device {ema_frequency_counter.device} != x device {x.device}"
        acts, non_zero_indices = self.encode(x, feature_indices)
        if method == 'with_acts':
            return acts

        x_reconstruct = self.decode(acts, non_zero_indices, feature_indices)

        if method == 'with_loss':
            l2_loss = normalized_mse(x_reconstruct, x)
            aux_acts, indices = self.activation.forward_aux(acts, ema_frequency_counter)
            e = x - x_reconstruct
            
            e_hat = self.decode(aux_acts, indices, feature_indices)
            aux_k = normalized_mse(e_hat, e)
            if aux_k.isnan().any():
                print("aux_k is nan")
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

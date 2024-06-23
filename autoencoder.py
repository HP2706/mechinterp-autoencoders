import json
import inspect
from modal import method
from torch.optim import Optimizer
from tqdm import tqdm
from typing_extensions import Self
import os
from typing import Optional
import torch
from torch import Tensor, nn
from pydantic import BaseModel, model_validator, Field
from torch.nn import functional as F
from typing import Callable, Union, Literal, List, Any, Type, TypeVar
from abc import ABC, abstractmethod, abstractproperty
from training_config import AutoencoderTrainConfig
from utils import slice_weights, slice_biases
from _types import Loss_Method, Methods
import einops
from jaxtyping import Float, Int
from loss_and_stats import (
    compute_l0_norm,
    compute_mean_absolute_error,
    compute_l1_sparsity,
    compute_normalized_L1_loss,
    compute_did_fire,
    compute_normalized_mse,
    compute_avg_num_firing_per_neuron,
)
if torch.cuda.is_available():
    from kernels import TritonDecoder


AUTOENCODER_TYPES = Literal['autoencoder', 'gated_autoencoder', 'topk_autoencoder']
AUTOENCODER_CLASS = Union['GatedAutoEncoder', 'AutoEncoder', 'TopKAutoEncoder']

class BasicStats(BaseModel):
    loss: Tensor
    x_reconstruct: Tensor
    acts: Tensor
    acts_sum : Tensor = Field(..., description="the mean of the sum of the activations over the batch")
    l2_loss: Tensor = Field(..., description="mean squared error")
    l1_loss: Tensor = Field(..., description="mean absolute error")
    normalized_l1_loss: Tensor = Field(..., description="normalized L1 loss")
    l_sparsity: Optional[Tensor] = Field(..., description="the sum of the absolute values of the activations")
    l0_norm: Tensor = Field(..., description="average number of non-zero entries in the activations")
    weight_stats : Optional[dict[str, float]] = Field(
        default=None, 
        description="the norms and sums of all the weights"
    )
    did_fire : Tensor = Field(..., description="test with ones and zeros for whether metric for whether activation fired or not")
    avg_num_firing_per_neuron : Tensor 

    class Config:
        arbitrary_types_allowed = True

    def format_data(self):
        data = self.model_dump()
        data.pop('weight_stats')
        if self.weight_stats is not None:
            data.update(self.weight_stats)
        formatted_results = {}
        for key, value in data.items():
            if isinstance(value, Tensor):
                if value.shape == ():
                    formatted_results[key] = value.item()
            elif value is None:
                continue
            else:
                formatted_results[key] = value
        return formatted_results


class GatedAutoEncoderResult(BasicStats):
    L_aux: Tensor
    class Config:
        arbitrary_types_allowed = True

    def format_data(self):
        base_loss = super().format_data()  # Call the superclass method
        return {
            **base_loss,
            "L_aux": self.L_aux.item(), 
        }

class AutoencoderResult(BasicStats):
    #TODO might add fields
    class Config:
        arbitrary_types_allowed = True

    def format_data(self):
        return super().format_data()

class AutoEncoderBaseConfig(BaseModel):
    type: AUTOENCODER_TYPES
    dict_mult: int
    d_input: int 
    l1_coeff: float
    seed: int = 42
    enc_dtype: Literal['fp32', 'fp16'] = 'fp16'
    device: Literal['cuda', 'mps', 'cpu'] = 'cuda'
    updated_anthropic_method : bool = True
    
    @property
    def d_sae(self):
        return self.d_input * self.dict_mult

    @property 
    def dtype(self):
        return torch.float32 if self.enc_dtype == 'fp32' else torch.float16
    
    @property
    def folder_name(self) -> str:
        return f'{self.type}_d_hidden_{self.d_input * self.dict_mult}_dict_mult_{self.dict_mult}'

    def create_folder_name(
        self, 
        train_cfg : AutoencoderTrainConfig
    ) -> str:
        with_ramp_str = '_with_l1_coeff_ramp' if train_cfg.with_ramp else ''
        with_loss_func = f'_{train_cfg.loss_func}'
        return (
            f'{self.folder_name}_'
            f'lr_{train_cfg.lr}_'
            f'steps_{train_cfg.n_steps}{with_ramp_str}{with_loss_func}'
        )

T = TypeVar('T', bound='AutoEncoderBase')

class AutoEncoderBase(nn.Module, ABC):
    def __init__(self, cfg : AutoEncoderBaseConfig):
        super().__init__()
        self.cfg = cfg

    def initialize_weights(self):
        '''
        initializes the following weights
        W_dec : d_hidden x d_input
        W_enc : d_input x d_hidden
        b_enc : d_hidden
        pre_bias : d_input
        '''
        d_hidden = self.cfg.d_input * self.cfg.dict_mult
        self.W_enc = nn.Parameter(
            torch.empty(self.cfg.d_input, d_hidden, dtype=self.cfg.dtype)
        )

        self.W_dec = nn.Parameter(torch.empty(d_hidden, self.cfg.d_input, dtype=self.cfg.dtype))
        nn.init.kaiming_uniform_(self.W_dec)
        self.set_decoder_norm_to_unit_norm()
        self.W_enc.data = self.W_dec.data.t().clone()
        self.pre_bias = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=self.cfg.dtype)) # initialize to zero
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype)) # initialize to zero
        
        assert not torch.isnan(self.W_dec).any(), f'self.W_dec contains nan: {self.W_dec.data}'
        assert not torch.isnan(self.W_enc).any(), f'self.W_enc contains nan: {self.W_enc.data}'

    @abstractmethod
    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):...

    @property
    def d_sae(self):
        return self.cfg.d_sae
    
    @abstractmethod
    def forward(self, x: Tensor, method: Methods) -> Any:
        """
        Abstract forward method that must be implemented by subclasses.
        The method parameter can dictate the behavior of the forward pass.
        """
        pass

    @torch.no_grad()
    def get_weight_data(self) -> dict[str, float]:
        weight_data = {}
        for name, value in inspect.getmembers(self):
            if isinstance(value, Union[nn.Parameter, nn.Linear]):
                weight_data[f'{name}_norm'] = torch.norm(value.data).item()
        return weight_data

    @method()
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @method()
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()
    
    @property
    def name(self):
        return self.__class__.__name__.lower()
    
    @classmethod
    def train_cfg_path_name(cls) -> str: 
        return 'config.json'

    @classmethod
    def model_cfg_path_name(cls) -> str: 
        return 'model_cfg.json'

    @classmethod
    def model_weights_path_name(cls) -> str: 
        return 'model.pt'


    @classmethod
    def load_from_checkpoint(
        cls, 
        dir_path : str, 
    ) -> tuple[AUTOENCODER_CLASS, AutoencoderTrainConfig]:
        """
        Loads the saved autoencoder from a checkpoint.
        If a json file is not provided, it is expected that the checkpoint is from a run of the autoencoder
        and the json file is assumed to be in the same directory as the checkpoint with the same name, but with
        a .json extension.
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"Got path to file {dir_path} not folder")

        print("property", repr(cls.train_cfg_path_name()))

        cfg_path = f"{dir_path}/{cls.train_cfg_path_name()}"
        print('cfg_path', cfg_path)
        train_cfg = AutoencoderTrainConfig.model_validate(
            json.load(open(cfg_path))
        )

        model_cfg_path = f"{dir_path}/{str(cls.model_cfg_path_name())}"
        print('model_cfg_path', model_cfg_path)
        model_cfg = json.load(open(model_cfg_path))
        
        class_name = cls.__name__
        if class_name in ['AutoEncoder', 'GatedAutoEncoder']:
            model_cfg = AutoEncoderBaseConfig.model_validate(model_cfg)
            if class_name == 'AutoEncoder':
                model = AutoEncoder(model_cfg) 
            else:
                model = GatedAutoEncoder(model_cfg)
        elif class_name == 'TopKAutoEncoder':
            model_cfg = TopKAutoEncoderConfig.model_validate(model_cfg)
            model = TopKAutoEncoder(model_cfg)
        else:
            raise ValueError(f"Config type '{class_name}' does not match the expected type '{cls.__name__.lower()}'")
        model.load_state_dict(torch.load(f'{dir_path}/{cls.model_weights_path_name()}'))
        return model, train_cfg
    
    def save_model(
        self,
        train_cfg: AutoencoderTrainConfig,
        model_dir : str
    ):
        name = self.cfg.create_folder_name(train_cfg)
        self.model_path = f'{model_dir}/{name}'
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(
            self.state_dict(), 
            f"{self.model_path}/{self.model_weights_path_name()}"
        )
        with open(f"{self.model_path}/{self.train_cfg_path_name()}", "w") as f:
            f.write(train_cfg.model_dump_json())
        with open(f"{self.model_path}/{self.model_cfg_path_name()}", "w") as f:
            f.write(self.cfg.model_dump_json())

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj


class AutoEncoder(AutoEncoderBase):
    def __init__(self, cfg : AutoEncoderBaseConfig):
        super().__init__(cfg)
        self.d_hidden = cfg.d_input * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.initialize_weights()
        self.activation = nn.ReLU()
        self.l1_coeff = cfg.l1_coeff
        self.to(self.cfg.device) # move to device
    
    def _prepare_weights(self, feature_indices: Optional[slice]):
        W_enc = slice_weights(self.W_enc, feature_indices)
        pre_bias = slice_biases(self.pre_bias, feature_indices)
        W_dec = slice_weights(self.W_dec, feature_indices)
        b_enc = slice_biases(self.b_enc, feature_indices)
        return W_enc, pre_bias, W_dec, b_enc

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, AutoencoderResult]:
        W_enc, pre_bias, W_dec, b_enc = self._prepare_weights(feature_indices)
        x_center = x - pre_bias
        acts = self.activation(x_center @ W_enc + b_enc)
        if method == 'with_acts':
            return acts

        x_reconstruct = acts @ W_dec + pre_bias
        if method in ['with_loss', 'with_new_loss']:
            l2_loss = compute_normalized_L1_loss(x_reconstruct, x)
            if method == 'with_loss':
                L_sparse = compute_l1_sparsity(acts)
            else:  # method == 'with_new_loss'
                L_sparse = (acts.abs() * torch.norm(W_dec, dim=-1)).sum()  # alternative sparsity metric
            
            loss = l2_loss + (self.l1_coeff * L_sparse)
            
            return AutoencoderResult(
                loss=loss, 
                x_reconstruct=x_reconstruct, 
                acts=acts, 
                acts_sum=acts.sum(1).mean(),
                l2_loss=l2_loss, 
                l_sparsity=L_sparse,
                l1_loss=compute_mean_absolute_error(x_reconstruct, x),
                normalized_l1_loss=compute_normalized_L1_loss(acts, x),
                l0_norm=compute_l0_norm(acts),
                did_fire=compute_did_fire(acts),
                avg_num_firing_per_neuron=compute_avg_num_firing_per_neuron(acts)
            )
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

#from paper https://arxiv.org/pdf/2404.16014
class GatedAutoEncoder(AutoEncoderBase):
    def __init__(self, cfg : AutoEncoderBaseConfig):
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

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, GatedAutoEncoderResult]:

        W_gate = slice_weights(self.W_gate, feature_indices)
        b_gate = slice_biases(self.b_gate, feature_indices)
        W_mag = slice_weights(self.W_mag, feature_indices)
        b_mag = slice_biases(self.b_mag, feature_indices)
        W_dec = slice_weights(self.W_dec, feature_indices)
        pre_bias = slice_biases(self.pre_bias, feature_indices)
        x = x[:, feature_indices] if feature_indices is not None else x

        x_center = x - pre_bias
        gate_center = x_center @ W_gate + b_gate
        active_features = (gate_center > 0).to(self.cfg.dtype)
        feature_magnitudes = self.activation(x_center @ W_mag + b_mag)
        # Computing final activations
        acts = active_features * feature_magnitudes
        # Reconstructing x
        x_reconstruct = acts @ W_dec + pre_bias

        if method == 'with_acts':
            return acts
        elif method == 'reconstruct':
            return x_reconstruct
        elif method == 'with_loss':
            # Reconstruct loss
            l2 = compute_normalized_mse(x_reconstruct, x)
            # Computing Sparsity loss
            via_gate_feature_magnitudes = self.activation(gate_center)
            L_Sparsity = via_gate_feature_magnitudes.float().sum()

            # Computing L_aux with frozen decoder
            with torch.no_grad():
                W_dec_frozen = W_dec.detach()
                b_enc_frozen = b_mag.detach()
            via_gate_reconstruction = (via_gate_feature_magnitudes @ W_dec_frozen + b_enc_frozen)
            L_aux = compute_normalized_mse(via_gate_reconstruction, x)
            loss = l2 + (self.l1_coeff * L_Sparsity) + L_aux

            return GatedAutoEncoderResult(
                loss=loss, 
                l2_loss=l2,
                l1_loss=compute_mean_absolute_error(x_reconstruct, x),
                normalized_l1_loss=compute_normalized_L1_loss(acts, x),
                l_sparsity=L_Sparsity,
                l0_norm=compute_l0_norm(acts),
                x_reconstruct=x_reconstruct, 
                acts=acts, 
                acts_sum=acts.sum(),
                L_aux=L_aux,
                did_fire=compute_did_fire(acts),
                avg_num_firing_per_neuron=compute_avg_num_firing_per_neuron(acts),
                # weight_stats=self.get_weight_data()
            )
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
        
#based on https://cdn.openai.com/papers/sparse-autoencoders.pdf
class TopKActivation(nn.Module):
    def __init__(
        self, 
        k: int, 
        k_aux : int, 
    ) -> None:
        super().__init__()
        self.k = k
        self.k_aux = k_aux

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        #we assume the relu already has been applied
        topk = torch.topk(x, k=self.k, dim=-1)
        result = torch.zeros_like(x, device=x.device)
        result.scatter_(-1, topk.indices, topk.values)
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

class TopKAutoEncoderConfig(AutoEncoderBaseConfig):
    k: int
    k_aux: int 
    # select aux_k as a power of two as close to d_model/2 as possible 
    # from openai paper https://cdn.openai.com/papers/sparse-autoencoders.pdf

    @property
    def folder_name(self) -> str:
        super_folder_name = super().folder_name
        return f"{super_folder_name}_k_{self.k}_k_aux_{self.k_aux}"
    
   
class TopKAutoEncoder(AutoEncoderBase):
    def __init__(
        self, 
        model_cfg: TopKAutoEncoderConfig
    ) -> None:
        super().__init__(model_cfg)
        self.cfg = model_cfg
        self.d_hidden = model_cfg.d_input
        self.activation = TopKActivation(k=model_cfg.k, k_aux=model_cfg.k_aux)
        self.initialize_weights()#initialize weights
        self.set_decoder_norm_to_unit_norm()
        

    def encode(
        self, 
        x: Float[Tensor, "... d_in"]
    ) -> tuple[
            Float[Tensor, "... d_sae"], 
            Int[Tensor, "... d_sae"]
        ]:
        # Remove decoder bias as per Anthropic
        sae_in = torch.relu(x.to(self.cfg.dtype) - self.pre_bias)
        return self.activation.forward(sae_in @ self.W_enc + self.b_enc)

    def decode(self, acts: Float[Tensor, "... d_sae"])-> Float[Tensor, "... d_in"]:
        return acts @ self.W_dec + self.pre_bias

    def decode_kernel(
        self,
        top_acts: Float[Tensor, "... d_sae"],
        top_indices: Int[Tensor, "..."],
    ) -> Float[Tensor, "... d_in"]:
        y = TritonDecoder.apply(top_indices, top_acts.to(self.cfg.dtype), self.W_dec.mT)
        return y + self.pre_bias

    def forward2(self, x: Tensor, dead_mask: Tensor | None = None)-> Float[Tensor, "... d_in"]:
        latent_acts = self.encode(x)
        top_acts, top_indices = latent_acts.topk(self.cfg.k, sorted=False)

        # Decode and compute residual
        sae_out = self.decode_kernel(top_acts, top_indices)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], latent_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode_kernel(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum(0)
            auxk_loss = scale * torch.mean(auxk_loss / total_variance)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum(0)
        fvu = torch.mean(l2_loss / total_variance)
        return sae_out

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        ema_frequency_counter : Tensor,
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, AutoencoderResult]:
        assert ema_frequency_counter.device == x.device, f"ema_frequency_counter device {ema_frequency_counter.device} != x device {x.device}"

        x_center = x - self.pre_bias
        acts = x_center @ self.W_enc + self.b_enc
        if method == 'with_acts':
            return acts
        
        acts_top_k = self.activation(acts)

        x_reconstruct = acts_top_k @ self.W_dec + self.pre_bias
        if method == 'with_loss':
            l2_loss = compute_normalized_mse(x_reconstruct, x)
            aux_acts = self.activation.forward_aux(acts, ema_frequency_counter)
            e = x - x_reconstruct
            e_hat = aux_acts @ self.W_dec + self.pre_bias
            aux_k = compute_normalized_mse(e_hat, e)
            if aux_k.isnan().any():
                print("aux_k is nan")
                aux_k = torch.zeros_like(aux_k)
            alpha = 1/32 #as specified in openai autoencoder paper
            loss = l2_loss + alpha * aux_k

            return AutoencoderResult(
                loss=loss, 
                x_reconstruct=x_reconstruct, 
                acts=acts_top_k, 
                acts_sum=acts_top_k.sum(1).mean(),
                l2_loss=loss, 
                l_sparsity=None, #sparsity loss,
                l1_loss=compute_mean_absolute_error(x_reconstruct, x),
                normalized_l1_loss=compute_normalized_L1_loss(acts_top_k, x),
                l0_norm=compute_l0_norm(acts_top_k),
                did_fire=compute_did_fire(acts_top_k),
                avg_num_firing_per_neuron=compute_avg_num_firing_per_neuron(acts_top_k)
            )
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
        
""" 
class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        d_sae = d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, d_sae, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        self.encoder.weight.data *= 0.1    # Small init means FVU starts below 1.0

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @property
    def device(self):
        return self.b_dec.device

    @property
    def dtype(self):
        return self.b_dec.dtype """


class EMA:
    def __init__(self, model, ema_decay = 0.999):
        self.ema_decay = ema_decay
        self.model = model
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
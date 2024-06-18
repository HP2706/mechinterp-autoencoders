import json
import math
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
from utils import slice_weights, slice_biases
from typing import overload
from _types import Loss_Method, Methods
from loss_and_stats import (
    compute_l0_norm,
    compute_mean_absolute_error,
    compute_l1_sparsity,
    compute_normalized_L1_loss,
    compute_did_fire,
    compute_normalized_mse,
    compute_avg_num_firing_per_neuron,
)

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

AUTOENCODER_TYPES = Literal['autoencoder', 'gated_autoencoder', 'topk_autoencoder']


class AutoencoderModelConfig(BaseModel):
    type: AUTOENCODER_TYPES
    dict_mult: int
    d_mlp: int 
    l1_coeff: float
    seed: int = 42
    enc_dtype: Literal['fp32', 'fp16'] = 'fp16'
    device: Literal['cuda', 'mps', 'cpu'] = 'cuda'
    updated_anthropic_method : bool = True
    
    @property
    def d_sae(self):
        return self.d_mlp * self.dict_mult

    @property 
    def dtype(self):
        return torch.float32 if self.enc_dtype == 'fp32' else torch.float16
    
    @property
    def folder_name(self):
        return f'{self.type}_d_hidden_{self.d_mlp * self.dict_mult}_dict_mult_{self.dict_mult}'

class AutoencoderTrainConfig(AutoencoderModelConfig):
    wandb_log : bool = True
    normalize_w_dec : bool = Field(...,description="""
        normalize the wdec to unit norm after each step necessary to 
        avoid gameability of sparsity metric
        """
    )
    type: AUTOENCODER_TYPES
    batch_size: int
    buffer_mult: int
    num_tokens: Optional[int] = None
    l1_coeff: float
    with_ramp : bool = False
    lr : float
    beta1: float
    beta2: float
    seq_len: int
    batch_size: int
    buffer_size: int
    buffer_batches: int
    n_epochs: int
    n_steps: int
    training_set : Optional[List[str]] = None
    validation_set : Optional[List[str]] = None
    loss_func : Optional[Loss_Method] = 'with_loss'
    anthropic_resampling : bool = False
    anthropic_resample_look_back_steps : Optional[int] = None
    sched_lr_factor : Optional[float] = None

    @model_validator(mode='after')
    def check_loss_func(self, data)-> Self:
        if self.loss_func == 'with_new_loss' and self.type == 'gated_autoencoder':
            raise ValueError("Gated Autoencoder does not support 'with_new_loss' loss function")
        return data

    def create_basename(self)->str:
        with_ramp_str = '_with_l1_coeff_ramp' if self.with_ramp else ''
        with_loss_func = f'_{self.loss_func}'
        return (
            f'{self.folder_name}_'
            f'lr_{self.lr}_'
            f'steps_{self.n_steps}{with_ramp_str}{with_loss_func}'
        ) # group string for readability


    @classmethod
    def default(cls)-> 'AutoencoderTrainConfig':
        return cls(
            batch_size=1,
            buffer_mult=1,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            seq_len=1,
            buffer_size=1,
            buffer_batches=1,
            n_epochs=1,
            n_steps=1,
            type='autoencoder',
            loss_func='with_loss',
            dict_mult=2,
            d_mlp=768,
            l1_coeff=0.1,
            seed=0,
            normalize_w_dec=True,
            enc_dtype='fp32',
            device='cuda'
        )

T = TypeVar('T', bound='AutoEncoderBase')

class AutoEncoderBase(nn.Module, ABC):
    def __init__(self, cfg : AutoencoderModelConfig):
        super().__init__()
        self.cfg = cfg

    def initialize_weights(self):
        d_hidden = self.cfg.d_mlp * self.cfg.dict_mult
        self.W_enc = nn.Parameter(
            torch.empty(self.cfg.d_mlp, d_hidden, dtype=self.cfg.dtype)
        )

        self.W_dec = nn.Parameter(torch.empty(d_hidden, self.cfg.d_mlp, dtype=self.cfg.dtype))
        nn.init.kaiming_uniform_(self.W_dec)

        with torch.no_grad():
            #we normalize to 
            if not self.cfg.updated_anthropic_method:
                #in the first autoencoder we constrain the W_DEC to unit norm     
                #initialization as described in https://transformer-circuits.pub/2024/april-update/index.html#training-saes   
                self.W_dec.data /= torch.norm(self.W_dec.data, dim=0, keepdim=True)
            else:
                self.W_dec.data *= 0.1 / torch.norm(self.W_dec.data, dim=0, keepdim=True)
                # Initialize W_enc to self.W_dec^T
            
        assert not torch.isnan(self.W_dec).any(), f'self.W_dec contains nan after normalization: {self.W_dec.data}'
        self.W_enc.data = self.W_dec.data.t().clone()
        assert not torch.isnan(self.W_enc).any(), f'self.W_enc contains nan after transposition: {self.W_enc.data}'

        self.pre_bias = nn.Parameter(torch.zeros(self.d_in, dtype=self.cfg.dtype)) # initialize to zero
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype)) # initialize to zero
        
        assert not torch.isnan(self.W_dec).any(), f'self.W_dec contains nan: {self.W_dec.data}'
        assert not torch.isnan(self.W_enc).any(), f'self.W_enc contains nan: {self.W_enc.data}'

    @abstractmethod
    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):...

    @property
    def d_sae(self):
        return self.cfg.d_sae
    
    @abstractproperty
    def d_in(self):...

    @abstractmethod
    def forward(self, x: Tensor, method: Methods) -> Any:
        """
        Abstract forward method that must be implemented by subclasses.
        The method parameter can dictate the behavior of the forward pass.
        """
        pass

    @torch.no_grad()
    @abstractmethod
    def get_weight_data(self)-> dict[str, float]:
        pass

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()
    
    @property
    def name(self):
        return self.__class__.__name__.lower()
    
    @property
    def get_file_path(self):
        return self.metadata_cfg
    
    @classmethod
    def load_from_checkpoint(
        cls, 
        dir_path : str, 
    ) -> Union['GatedAutoEncoder', 'AutoEncoder']:
        """
        Loads the saved autoencoder from a checkpoint.
        If a json file is not provided, it is expected that the checkpoint is from a run of the autoencoder
        and the json file is assumed to be in the same directory as the checkpoint with the same name, but with
        a .json extension.
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"Got path to file {dir_path} not folder")
            
        json_path = f'{dir_path}/config.json'
        
        if not os.path.exists(json_path): 
            raise ValueError("no corresponding json file found for the checkpoint. a json file should be provided")

        cls.dir_name = dir_path.split('/')[-1] 
        cfg = AutoencoderTrainConfig(**json.loads(open(json_path).read()))
        cls.metadata_cfg = cfg


        if cfg.type == 'autoencoder':
            model = AutoEncoder(cfg)
        elif cfg.type == 'gated_autoencoder':
            model = GatedAutoEncoder(cfg)
        else:
            raise ValueError(f"Config type '{cfg.type}' does not match the expected type '{cls.__name__.lower()}'")

        model.load_state_dict(torch.load(f'{dir_path}/model.pt', map_location=cfg.device))
        return model

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    @classmethod
    @abstractmethod
    def default(cls: Type[T]) -> T:
        """Should be implemented to return an instance of the calling class."""
        pass
    #inspired by neel nanda https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=qCF9odNdAvKX

class AutoEncoder(AutoEncoderBase):
    def __init__(self, cfg : AutoencoderModelConfig):
        super().__init__(cfg)
        self.d_hidden = cfg.d_mlp * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.initialize_weights()
        self.activation = nn.ReLU()
        self.l1_coeff = cfg.l1_coeff
        self.to(self.cfg.device) # move to device

    @property
    def d_in(self):
        return self.W_enc.shape[0]

    @classmethod
    def default(cls: Type[T]) -> T:
        return cls(AutoencoderTrainConfig.default())

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str,
    ) -> 'AutoEncoder': ...

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
    ) -> 'AutoEncoder': ...

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
    ) -> 'AutoEncoder': ...

    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
    ) -> 'AutoEncoder': 
        return super(AutoEncoder, cls).load_from_checkpoint(checkpoint_path) #type: ignore

    def get_weight_data(self)-> dict[str, float]:
        return {
            'W_enc_norm': torch.norm(self.W_enc).item(),
            'pre_bias_norm': torch.norm(self.pre_bias).item(),
            'W_dec_norm': torch.norm(self.W_dec).item(),
            'b_enc_norm': torch.norm(self.b_enc).item(),
        }
    
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
                    assert k.data.shape == (self.d_sae, self.d_in), f"expected shape (self.d_sae, self.d_in) got {k.data.shape}"
                    v[v_key][indices, :] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_in, self.d_sae), f"expected shape (self.d_in,) got {k.data.shape}"
                    v[v_key][:, indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae,), f"expected shape (self.d_in, self.d_sae) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,), f"expected shape (self.d_sae,) got {k.data.shape}"
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")

#from paper https://arxiv.org/pdf/2404.16014
class GatedAutoEncoder(AutoEncoderBase):
    def __init__(self, cfg : AutoencoderModelConfig):
        super().__init__(cfg)
        d_hidden = cfg.d_mlp * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        # Initialize parameters for the gated autoencoder
        self.l1_coeff = cfg.l1_coeff
        self.W_gate = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, d_hidden, dtype=cfg.dtype)))

        self.b_gate = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))
        self.b_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))

        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype)) # initialize to zero

        # Initialize W_mag as a vector of learnable parameters
        self.r_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) #TODO should this be initialized to zero?
        weight_init = torch.exp(self.r_mag)[None, :] * self.W_gate
        self.W_mag = nn.Parameter(weight_init)
        self.activation = nn.ReLU()
        self.to(self.cfg.device)

    @property
    def d_in(self):
        return self.W_gate.shape[0]

    @classmethod
    def default(cls: Type[T]) -> T:
        return cls(AutoencoderTrainConfig.default())

    @overload
    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
    ) -> 'GatedAutoEncoder': ...

    @overload
    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        json_path : Optional[str] = None
    ) -> 'GatedAutoEncoder': ...

    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path: str, 
        json_path: Optional[str] = None
    ) -> 'GatedAutoEncoder':
        # Implementation remains the same
        return super(GatedAutoEncoder, cls).load_from_checkpoint(checkpoint_path, json_path) #type: ignore

    def get_weight_data(self)-> dict[str, float]:
        return {
            'W_gate_norm': torch.norm(self.W_gate).item(),
            'b_gate_norm': torch.norm(self.b_gate).item(),
            'W_mag_norm': torch.norm(self.W_mag).item(),
            'b_mag_norm': torch.norm(self.b_mag).item(),
            'W_dec_norm': torch.norm(self.W_dec).item(),
            'b_enc_norm': torch.norm(self.b_enc).item(),
        }

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
                    assert k.data.shape == (self.d_in, self.d_hidden), f"expected shape ({self.d_in}, {self.d_hidden}) got {k.data.shape}"
                    v[v_key][:, indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,), f"expected shape (self.d_sae,) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae,), f"expected shape (self.d_sae,) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_sae, self.d_in), f"expected shape (self.d_sae, self.d_in) got {k.data.shape}"
                    v[v_key][indices, :] = 0.0
                elif dict_idx == 4:
                    assert k.data.shape == (self.d_in,), f"expected shape (self.d_in,) got {k.data.shape}"
                elif dict_idx == 5:
                    assert k.data.shape == (self.d_in, self.d_sae), f"expected shape (self.d_in, 1536) got {k.data.shape}"
                    v[v_key][:, indices]
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
        
#based on https://cdn.openai.com/papers/sparse-autoencoders.pdf
class TopKActivation(nn.Module):
    def __init__(self, k: int, k_aux : int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.k_aux = k_aux
        self.postact_fn = postact_fn

    def forward(self, x: Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x, device=x.device)
        result.scatter_(-1, topk.indices, values)
        return result
    
    def forward_aux(self, x: Tensor, ema_frequency_counter: Tensor) -> Tensor:
        topk = torch.topk(
            ema_frequency_counter, 
            k=self.k_aux, 
            dim=-1, 
        )
        dead_features = x[:, topk.indices] 
        values = self.postact_fn(dead_features)
        result = torch.zeros_like(x, device=x.device)
        result.scatter_(-1, topk.indices.unsqueeze(0).expand(x.size(0), -1), values)
        return result

class TopKAutoEncoderModelConfig(AutoencoderModelConfig):
    k: int
    k_aux: int 
    # select aux_k as a power of two as close to d_model/2 as possible 
    # from openai paper https://cdn.openai.com/papers/sparse-autoencoders.pdf

    @property
    def folder_name(self) -> str:
        super_folder_name = super().folder_name
        return f"{super_folder_name}_k_{self.k}_k_aux_{self.k_aux}"

class TopKAutoEncoder(nn.Module):
    def __init__(
        self, model_cfg: TopKAutoEncoderModelConfig
    ) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.d_hidden = model_cfg.d_mlp
        d_mlp = model_cfg.d_mlp
        device = model_cfg.device
        self.W_enc = nn.Parameter(torch.randn(d_mlp, self.d_hidden, device=device))
        self.pre_bias = nn.Parameter(torch.zeros(self.d_hidden, device=device))
        self.W_dec = nn.Parameter(torch.randn(self.d_hidden, d_mlp, device=device))
        self.latent_bias = nn.Parameter(torch.zeros(d_mlp, device=device))

        nans = torch.isnan(self.W_dec.data)
        assert not nans.any(), f'self.W_dec contains nan pre normalization: {self.W_dec.data}, nans: {nans}'
        self.activation = TopKActivation(k=model_cfg.k, k_aux=model_cfg.k_aux)
        with torch.no_grad():
            #we normalize to 
            eps = 1
            norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
            if torch.any(norm == 0):
                raise ValueError("Zero norm encountered in W_dec")
            
            #we initialize the weights to have a fixed L2 norm of 0.1
            norm = norm + eps
            self.W_dec.data *= 0.1 / (norm + eps)
            print(self.W_dec.data, "norm", torch.norm(self.W_dec.data))
            # Initialize W_enc to self.W_dec^T
            
        #we set w_enc to transpose of w_dec
        nans = torch.isnan(self.W_dec.data)
        assert not nans.any(), f'self.W_dec contains nan post normalization: {self.W_dec.data}, nans: {nans}'
    
    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct'],
        ema_frequency_counter : Tensor,
        feature_indices: Optional[slice] = None
    ) -> Union[Tensor, AutoencoderResult]:
        assert ema_frequency_counter.device == x.device, f"ema_frequency_counter device {ema_frequency_counter.device} != x device {x.device}"

        x_center = x - self.pre_bias
        acts = x_center @ self.W_enc + self.latent_bias
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
import json
import math
from torch.optim import Optimizer
from tqdm import tqdm
from typing_extensions import Self
import os
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from pydantic import BaseModel, model_validator, Field
from torch.nn import functional as F
from typing import Callable, Union, Literal, List, Any, Type, TypeVar
from abc import ABC, abstractmethod, abstractproperty
from utils import get_device
from typing import overload
from _types import Loss_Method, Methods
from loss_and_stats import (
    compute_l0_norm,
    compute_mean_absolute_error,
    compute_mse,
    compute_l1_sparsity,
    compute_normalized_mse,
    compute_normalized_L1_loss,
    compute_did_fire,
    compute_mean_firing_percentage,
    compute_avg_num_firing_per_neuron,
    compute_mean_firing_per_batch
)

class BasicStats(BaseModel):
    loss: Tensor
    x_reconstruct: Tensor
    acts: Tensor
    acts_sum : Tensor = Field(..., description="the mean of the sum of the activations over the batch")
    l2_loss: Tensor = Field(..., description="mean squared error")
    l1_loss: Tensor = Field(..., description="mean absolute error")
    normalized_l1_loss: Tensor = Field(..., description="normalized L1 loss")
    l_sparsity: Tensor = Field(..., description="the sum of the absolute values of the activations")
    l0_norm: Tensor = Field(..., description="average number of non-zero entries in the activations")
    normalized_mse : Tensor 
    weight_stats : Optional[dict[str, float]] = Field(
        default=None, 
        description="the norms and sums of all the weights"
    )
    mean_firing_percentage : Tensor = Field(..., description="the mean firing percentage of the neurons across the batch")

    did_fire : Tensor = Field(..., description="test with ones and zeros for whether metric for whether activation fired or not")
    avg_num_firing : Tensor = Field(..., description="average number of times a neuron fires")

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
    #loss, x_reconstruct, acts, L_Reconstruct, L_Sparsity, L_aux
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

class AutoencoderModelConfig(BaseModel):
    type: Literal['autoencoder', 'gated_autoencoder']
    dict_mult: int
    d_mlp: int 
    l1_coeff: float
    seed: int = 42
    enc_dtype: Literal['fp32', 'fp16'] = 'fp16'
    device: Literal['cuda', 'mps'] = 'cuda'
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

class AutoencoderConfig(AutoencoderModelConfig):
    type: Literal['autoencoder', 'gated_autoencoder']
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
    def default(cls)-> 'AutoencoderConfig':
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
        self.W_enc = nn.Parameter(torch.empty(self.cfg.d_mlp,d_hidden, dtype=self.cfg.dtype))
        torch.nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5)) #TODO why sqrt(5)
        self.W_dec = nn.Parameter(self.W_enc.data.t().clone())
        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            if not self.cfg.updated_anthropic_method:
                #in the first autoencoder we constrain the W_DEC to unit norm     
                #initialization as described in https://transformer-circuits.pub/2024/april-update/index.html#training-saes   
                self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
            else:
                #we initialize the weights to have a fixed L2 norm of 0.1
                self.W_dec.data *= 0.1 / torch.norm(self.W_dec.data, dim=0, keepdim=True)
                # Initialize W_enc to self.W_dec^T
        torch.nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        self.W_dec.data = self.W_enc.data.t().clone()

        with torch.no_grad():
            if not self.cfg.updated_anthropic_method:
                self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
            else:
                self.W_dec.data *= 0.1 / torch.norm(self.W_dec.data, dim=0, keepdim=True)

        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, dtype=self.cfg.dtype)) # initialize to zero
        self.b_dec = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=self.cfg.dtype)) # initialize to zero


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
    def get_single_feature_acts(
        self,     
        model_acts : torch.Tensor, 
        feature_index : int,
    ) -> torch.Tensor:
        """
        Returns the activations of a single feature.
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
        cfg = AutoencoderConfig(**json.loads(open(json_path).read()))
        cls.metadata_cfg = cfg

        device = get_device()

        if cfg.type == 'autoencoder':
            model = AutoEncoder(cfg)
        elif cfg.type == 'gated_autoencoder':
            model = GatedAutoEncoder(cfg)
        else:
            raise ValueError(f"Config type '{cfg.type}' does not match the expected type '{cls.__name__.lower()}'")

        model.load_state_dict(torch.load(f'{dir_path}/model.pt', map_location=device))
        return model

    @abstractmethod
    @torch.no_grad()
    def decode(self, acts : Tensor) -> Tensor:
        pass

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
        self.relu = nn.ReLU()
       
        self.l1_coeff = cfg.l1_coeff
        self.device = get_device()

        self.to(self.device) # move to device

    @property
    def d_in(self):
        return self.W_enc.shape[0]

    @classmethod
    def default(cls: Type[T]) -> T:
        return cls(AutoencoderConfig.default())

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
            'W_enc_sum': self.W_enc.sum().item(),
            'b_enc_sum': self.b_enc.sum().item(),
            'W_dec_sum': self.W_dec.sum().item(),
            'b_dec_sum': self.b_dec.sum().item(),
            'W_enc_norm': torch.norm(self.W_enc).item(),
            'b_enc_norm': torch.norm(self.b_enc).item(),
            'W_dec_norm': torch.norm(self.W_dec).item(),
            'b_dec_norm': torch.norm(self.b_dec).item(),
        }

    def get_single_feature_acts(
        self,     
        model_acts : torch.Tensor, 
        feature_index : int,
    ) -> torch.Tensor:
        '''gets the activation values for a specific feature index(index of the hidden layer)'''

        if model_acts.device != self.cfg.device:
            model_acts = model_acts.to(self.device)

        feature_in = self.W_enc[:, feature_index]
        feature_bias = self.b_enc[feature_index]
        feature_acts = F.relu((model_acts - self.b_dec) @ feature_in + feature_bias) # shape (batch, seq_len)
        feature_acts = feature_acts.cpu()
    
        return feature_acts.unsqueeze(-1) #[batch, 1]

    @overload
    def forward(self, x: Tensor, method: Literal['with_acts', 'reconstruct'], normalize_weights: bool = False) -> Tensor: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_loss'], normalize_weights: bool = False) -> AutoencoderResult: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_new_loss'], normalize_weights: bool = False) -> AutoencoderResult: ...

    def forward(
        self,
        x: Tensor, 
        method: Methods = 'with_loss',
        normalize_weights: bool = False
    ) -> Union[Tensor, AutoencoderResult]:

        W_enc = self.W_enc
        b_enc = self.b_enc
        W_dec = self.W_dec
        b_dec = self.b_dec

        if normalize_weights:
            #with torch.no_grad(): ?
            norm_factor = torch.norm(W_dec, dim=0, keepdim=True).mean()  # Reshape norm_factor for broadcasting
            W_enc = W_enc * norm_factor
            b_enc = b_enc * norm_factor  # Ensure b_enc is correctly broadcasted
            W_dec = W_dec / norm_factor  # Normalize W_dec directly

        x_center = x - b_dec
        acts = self.relu(x_center @ W_enc + b_enc)
        if method == 'with_acts':
            return acts
        
        x_reconstruct = acts @ W_dec + b_dec
        if method in ['with_loss', 'with_new_loss']:
            l2_loss = compute_mse(x_reconstruct, x)
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
                normalized_mse=compute_normalized_mse(x_reconstruct, x),
                did_fire=compute_did_fire(acts),
                mean_firing_percentage=compute_mean_firing_percentage(acts),
                avg_num_firing=compute_avg_num_firing_per_neuron(acts),
                #weight_stats=self.get_weight_data(),
            )
        elif method == 'reconstruct':
            return x_reconstruct
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def decode(self, acts : Tensor) -> Tensor:
        return acts @ self.W_dec + self.b_dec
    
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
        self.relu = nn.ReLU()
        # Initialize parameters for the gated autoencoder
        self.l1_coeff = cfg.l1_coeff
        self.W_gate = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, d_hidden, dtype=cfg.dtype)))

        self.b_gate = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))
        self.b_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype))

        self.W_dec = nn.Parameter()
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype)) # initialize to zero

        # Initialize W_mag as a vector of learnable parameters
        self.r_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) #TODO should this be initialized to zero?
        weight_init = torch.exp(self.r_mag)[None, :] * self.W_gate
        self.W_mag = nn.Parameter(weight_init)
        self.device = get_device()
        self.to(self.device)

    @property
    def d_in(self):
        return self.W_gate.shape[0]

    @classmethod
    def default(cls: Type[T]) -> T:
        return cls(AutoencoderConfig.default())

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

    def get_single_feature_acts(
        self,     
        model_acts : torch.Tensor, 
        feature_index : int,
    ) -> torch.Tensor:
        '''gets the activation values for a specific feature index (index of the hidden layer)'''

        if model_acts.device != self.device:
            model_acts = model_acts.to(self.device)

        model_acts_centered = model_acts - self.b_dec
        gate_activation = model_acts_centered @ self.W_gate[:, feature_index] + self.b_gate[feature_index]
        active_feature = (gate_activation > 0).float()
        feature_magnitude = self.relu(model_acts_centered @ self.W_mag[:, feature_index] + self.b_mag[feature_index])
        feature_activation = active_feature * feature_magnitude

        return feature_activation.unsqueeze(-1) #[batch, 1]

    def get_weight_data(self)-> dict[str, float]:
        return {
            'W_gate_sum': self.W_gate.sum().item(),
            'b_gate_sum': self.b_gate.sum().item(),
            'W_mag_sum': self.W_mag.sum().item(),
            'b_mag_sum': self.b_mag.sum().item(),
            'W_dec_sum': self.W_dec.sum().item(),
            'b_dec_sum': self.b_dec.sum().item(),
            'W_gate_norm': torch.norm(self.W_gate).item(),
            'b_gate_norm': torch.norm(self.b_gate).item(),
            'W_mag_norm': torch.norm(self.W_mag).item(),
            'b_mag_norm': torch.norm(self.b_mag).item(),
            'W_dec_norm': torch.norm(self.W_dec).item(),
            'b_dec_norm': torch.norm(self.b_dec).item(),
        }


    @overload
    def forward(self, x: Tensor, method: Literal['with_acts', 'reconstruct']) -> Tensor: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_loss']) -> GatedAutoEncoderResult: ...

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct']
    ) -> Union[Tensor, GatedAutoEncoderResult]:

        x_center = x - self.b_dec
        gate_center = x_center @ self.W_gate + self.b_gate
        active_features = (gate_center > 0).float()

        feature_magnitudes = self.relu(x_center @ self.W_mag + self.b_mag)

        # Computing final activations
        acts = active_features * feature_magnitudes

        # Reconstructing x
        x_reconstruct = acts @ self.W_dec + self.b_dec
 
        if method == 'with_acts':
            return acts
        elif method == 'reconstruct':
            return x_reconstruct
        elif method == 'with_loss':
            #Reconstruct loss
            l2 = compute_mse(x_reconstruct, x)
            # Computing Sparsity loss
            via_gate_feature_magnitudes = self.relu(gate_center)
            L_Sparsity = via_gate_feature_magnitudes.float().sum()

            # Computing L_aux with frozen decoder
            with torch.no_grad():
                W_dec_frozen = self.W_dec.detach()
                b_dec_frozen = self.b_dec.detach()
            via_gate_reconstruction = (via_gate_feature_magnitudes @ W_dec_frozen + b_dec_frozen)
            L_aux = compute_mse(x, via_gate_reconstruction)
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
                normalized_mse=compute_normalized_mse(x_reconstruct, x),
                did_fire=compute_did_fire(acts),
                mean_firing_percentage=compute_mean_firing_per_batch(acts),
                avg_num_firing=compute_avg_num_firing_per_neuron(acts),
                #weight_stats=self.get_weight_data()
            )
        else:
            return x_reconstruct
        

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
        
    def decode(self, acts : Tensor) -> Tensor:
        return acts @ self.W_dec + self.b_dec

#based on https://cdn.openai.com/papers/sparse-autoencoders.pdf
class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

class TopKAutoEncoderConfig(AutoencoderModelConfig):
    k: int

class TopKAutoEncoder(AutoEncoder):
    def __init__(self, cfg : TopKAutoEncoderConfig):
        super().__init__(cfg)
        self.topk = TopK(cfg.k, postact_fn=nn.ReLU())

    @overload
    def forward(self, x: Tensor, method: Literal['with_acts', 'reconstruct'], normalize_weights: bool = False) -> Tensor: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_loss'], normalize_weights: bool = False) -> AutoencoderResult: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_new_loss'], normalize_weights: bool = False) -> AutoencoderResult: ...

    def forward(
        self,
        x: Tensor, 
        method: Methods = 'with_loss',
        normalize_weights: bool = False
    ) -> Union[Tensor, AutoencoderResult]:

        W_enc = self.W_enc
        b_enc = self.b_enc
        W_dec = self.W_dec
        b_dec = self.b_dec

        if normalize_weights:
            #with torch.no_grad(): ?
            norm_factor = torch.norm(W_dec, dim=0, keepdim=True).mean()  # Reshape norm_factor for broadcasting
            W_enc = W_enc * norm_factor
            b_enc = b_enc * norm_factor  # Ensure b_enc is correctly broadcasted
            W_dec = W_dec / norm_factor  # Normalize W_dec directly

        x_center = x - b_dec
        acts = self.topk(x_center @ W_enc + b_enc)
        if method == 'with_acts':
            return acts
        
        x_reconstruct = acts @ W_dec + b_dec
        if method in ['with_loss', 'with_new_loss']:
            l2_loss = compute_mse(x_reconstruct, x)
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
                normalized_mse=compute_normalized_mse(x_reconstruct, x),
                did_fire=compute_did_fire(acts),
                mean_firing_percentage=compute_mean_firing_percentage(acts),
                avg_num_firing=compute_avg_num_firing_per_neuron(acts),
                #weight_stats=self.get_weight_data(),
            )
        elif method == 'reconstruct':
            return x_reconstruct
        else:
            raise ValueError(f"Invalid method: {method}")



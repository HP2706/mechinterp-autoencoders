import json
import os
import time 
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from pydantic import BaseModel, field_validator
from torch.nn import functional as F
from typing import Tuple, Union, Literal, List, Any
from abc import ABC, abstractmethod
from mechninterp_utils import utils
#internal imports
from mechninterp_utils import mean_ablate_hook
from typing import overload, Protocol


class GatedAutoEncoderResult(BaseModel):
    #loss, x_reconstruct, acts, L_Reconstruct, L_Sparsity, L_aux
    loss: Tensor
    x_reconstruct: Tensor
    acts: Tensor
    L_Reconstruct: Tensor
    L_Sparsity: Tensor
    L_aux: Tensor

    class Config:
        arbitrary_types_allowed = True

class AutoencoderResult(BaseModel):
    loss: Tensor
    x_reconstruct: Tensor
    acts: Tensor
    l2_loss: Tensor
    l1_loss: Tensor

    class Config:
        arbitrary_types_allowed = True

class AutoencoderModelConfig(BaseModel):
    type: Literal['autoencoder', 'gated_autoencoder']
    dict_mult: int
    d_mlp: int 
    l1_coeff: float
    seed: int
    enc_dtype: Literal['fp32', 'fp16'] = 'fp32'
    device: Literal['cuda', 'mps'] = 'cuda'
    
    @property 
    def dtype(self):
        return torch.float32 if self.enc_dtype == 'fp32' else torch.float16


class AutoencoderConfig(AutoencoderModelConfig):
    batch_size: int
    buffer_mult: int
    num_tokens: Optional[int] = None
    l1_coeff: float
    lr : float
    beta1: float
    beta2: float
    seq_len: int
    batch_size: int
    buffer_size: int
    buffer_batches: int
    n_epochs: int
    n_steps: Optional[int] = None
    training_set : Optional[List[str]] = None
    validation_set : Optional[List[str]] = None


    def basename(self, epoch: Optional[int] = None)->str:
        epoch = epoch if epoch is not None else self.n_epochs
        return f'{self.type}_d_hidden_{self.d_mlp * self.dict_mult}_lr_{self.lr}_dict_mult_{self.dict_mult}_epoch_{epoch}'


class AutoEncoderBase(nn.Module, ABC):
    def __init__(self, cfg : AutoencoderModelConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, x: Tensor, method: str) -> Any:
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

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()
    
    @property
    def name(self):
        return self.__class__.__name__.lower()
    
    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        device : Literal['cuda', 'cpu', 'mps'],
        json_path : Optional[str] = None
    ) -> Union['GatedAutoEncoder', 'AutoEncoder']:
        """
        Loads the saved autoencoder from a checkpoint.
        If a json file is not provided, it is expected that the checkpoint is from a run of the autoencoder
        and the json file is assumed to be in the same directory as the checkpoint with the same name, but with
        a .json extension.
        """

        
        if not checkpoint_path.endswith(".pt"):
            raise ValueError("checkpoint_path must end with .pt")
        if json_path is None:
            json_path = checkpoint_path.replace(".pt", ".json")
            if not os.path.exists(json_path):
                raise ValueError("no corresponding json file found for the checkpoint. a json file should be provided")

        cfg = AutoencoderConfig(**json.loads(open(json_path).read()))
        cls.metadata_cfg = cfg
        if cfg.type == cls.__name__.lower():  # Ensure the type matches the class name
            model = cls(cfg)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            return model # type: ignore
        else:
            raise ValueError(f"Config type '{cfg.type}' does not match the expected type '{cls.__name__.lower()}'")

#inspired by neel nanda https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=qCF9odNdAvKX
class AutoEncoder(AutoEncoderBase):
    def __init__(self, cfg : AutoencoderModelConfig):
        super().__init__(cfg)
        d_hidden = cfg.d_mlp * cfg.dict_mult
        torch.manual_seed(cfg.seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) # initialize to zero
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype)) # initialize to zero
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.d_hidden = d_hidden
        self.l1_coeff = cfg.l1_coeff

        self.to(cfg.device) # move to device

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str,
        device : Literal['cuda', 'cpu', 'mps'],
    ) -> 'AutoEncoder': ...

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        device : Literal['cuda', 'cpu', 'mps'],
        json_path : Optional[str] = None
    ) -> 'AutoEncoder': ...

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        device : Literal['cuda', 'cpu', 'mps'],
    ) -> 'AutoEncoder': ...

    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        device : Literal['cuda', 'cpu', 'mps'],
        json_path : Optional[str] = None
    ) -> 'AutoEncoder': 
        return super(AutoEncoder, cls).load_from_checkpoint(checkpoint_path, device, json_path) #type: ignore

    def get_single_feature_acts(
        self,     
        model_acts : torch.Tensor, 
        feature_index : int,
    ) -> torch.Tensor:
        '''gets the activation values for a specific feature index(index of the hidden layer)'''

        if model_acts.device != self.cfg.device:
            model_acts = model_acts.to(self.cfg.device)

        feature_in = self.W_enc[:, feature_index]
        feature_bias = self.b_enc[feature_index]
        feature_acts = F.relu((model_acts - self.b_dec) @ feature_in + feature_bias) # shape (batch, seq_len)
        feature_acts = feature_acts.cpu()
    
        return feature_acts

    @overload
    def forward(self, x: Tensor, method: Literal['with_acts', 'reconstruct']) -> Tensor: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_loss']) -> AutoencoderResult: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_new_loss']) -> AutoencoderResult: ...

    def forward(
        self,
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct', 'with_new_loss'] = 'with_loss'
    ) -> Union[Tensor, AutoencoderResult]:
        
        x_center = x - self.b_dec
        acts = F.relu(x_center @ self.W_enc + self.b_enc)
        if method == 'with_acts':
            return acts
        x_reconstruct = acts @ self.W_dec + self.b_dec
        if method == 'with_loss':
            t0 = time.time()
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            l1_loss = self.l1_coeff * (acts.float().abs().sum())
            loss = l2_loss + l1_loss
            return AutoencoderResult(
                loss=loss, 
                x_reconstruct=x_reconstruct, 
                acts=acts, 
                l2_loss=l2_loss, 
                l1_loss=l1_loss
            )
        elif method == 'reconstruct':
            return x_reconstruct
        elif method == 'with_new_loss':
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            l1_loss = torch.sum(
                torch.abs(acts) 
                * 
                torch.norm(self.W_dec, dim=0, p=2)
            )
            loss = l2_loss + l1_loss
            return AutoencoderResult(
                loss=loss, 
                x_reconstruct=x_reconstruct, 
                acts=acts, 
                l2_loss=l2_loss, 
                l1_loss=l1_loss
            )
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


        cfg : dict = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json") # type: ignore
        cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"] 
        cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"] 
        cfg["device"] = device 
        valid_cfg = AutoencoderModelConfig(**cfg) 
        self = cls(cfg=valid_cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True)) # type: ignore
        return self


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

        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype)) # initialize to zero

        # Initialize W_mag as a vector of learnable parameters
        self.r_mag = nn.Parameter(torch.zeros(d_hidden, dtype=cfg.dtype)) #TODO should this be initialized to to zero?
        self.W_mag = torch.exp(self.r_mag)[None, :] * self.W_gate
        self.W_mag = nn.Parameter(self.W_mag)

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        device : Literal['cuda', 'cpu', 'mps'],
    ) -> 'GatedAutoEncoder': ...

    @classmethod
    @overload
    def load_from_checkpoint(
        cls, 
        checkpoint_path : str, 
        device : Literal['cuda', 'cpu', 'mps'],
        json_path : Optional[str] = None
    ) -> 'GatedAutoEncoder': ...

    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path: str, 
        device: Literal['cuda', 'cpu', 'mps'],
        json_path: Optional[str] = None
    ) -> 'GatedAutoEncoder':
        # Implementation remains the same
        return super(GatedAutoEncoder, cls).load_from_checkpoint(checkpoint_path, device, json_path) #type: ignore


    def get_single_feature_acts(
        self,     
        model_acts : torch.Tensor, 
        feature_index : int,
    ) -> torch.Tensor:
        '''gets the activation values for a specific feature index (index of the hidden layer)'''

        if model_acts.device != self.cfg.device:
            model_acts = model_acts.to(self.cfg.device)

        model_acts_centered = model_acts - self.b_dec
        gate_activation = model_acts_centered @ self.W_gate[:, feature_index] + self.b_gate[feature_index]
        active_feature = (gate_activation > 0).float()
        feature_magnitude = self.relu(model_acts_centered @ self.W_mag[:, feature_index] + self.b_mag[feature_index])
        feature_activation = active_feature * feature_magnitude

        return feature_activation.unsqueeze(-1)  # Ensure it keeps the right dimensions

    

    @overload
    def forward(self, x: Tensor, method: Literal['with_acts', 'reconstruct']) -> Tensor: ...
    @overload
    def forward(self, x: Tensor, method: Literal['with_loss']) -> GatedAutoEncoderResult: ...

    def forward(
        self, 
        x: Tensor, 
        method: Literal['with_acts', 'with_loss', 'reconstruct']
    ) -> Union[Tensor, GatedAutoEncoderResult]:
        start_time = time.time()

        # Centering x
        x_center = x - self.b_dec
        center_time = time.time()

        # Computing gate activations
        gate_center = x_center @ self.W_gate + self.b_gate
        gate_time = time.time()

        # Computing active features
        active_features = (gate_center > 0).float()
        active_features_time = time.time()

        # Computing feature magnitudes
        feature_magnitudes = self.relu(x_center @ self.W_mag + self.b_mag)
        feature_magnitudes_time = time.time()

        # Computing final activations
        acts = active_features * feature_magnitudes
        acts_time = time.time()

        # Reconstructing x
        x_reconstruct = acts @ self.W_dec + self.b_dec
        reconstruct_time = time.time()

        total_time = time.time() - start_time
        print("distribution of time spent in each part of the forward pass")
        print(f"Time spent centering x: {(center_time - start_time) / total_time * 100:.2f}%")
        print(f"Time spent computing gate activations: {(gate_time - center_time) / total_time * 100:.2f}%")
        print(f"Time spent computing active features: {(active_features_time - gate_time) / total_time * 100:.2f}%")
        print(f"Time spent computing feature magnitudes: {(feature_magnitudes_time - active_features_time) / total_time * 100:.2f}%")
        print(f"Time spent computing final activations: {(acts_time - feature_magnitudes_time) / total_time * 100:.2f}%")
        print(f"Time spent reconstructing x: {(reconstruct_time - acts_time) / total_time * 100:.2f}%")

        if method == 'with_acts':
            print(f"Total time: {reconstruct_time - start_time:.6f}s")
            return acts
        elif method == 'reconstruct':
            print(f"Total time: {reconstruct_time - start_time:.6f}s")
            return x_reconstruct
        elif method == 'with_loss':
            loss_time = time.time()
            # Computing L_Reconstruct
            L_Reconstruct = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
            l_reconstruct_time = time.time()

            # Computing L_Sparsity
            via_gate_feature_magnitudes = self.relu(gate_center)
            L_Sparsity = self.l1_coeff * (via_gate_feature_magnitudes.float().sum())
            l_sparsity_time = time.time()

            # Computing L_aux with frozen decoder
            with torch.no_grad():
                W_dec_frozen = self.W_dec.detach()
                b_dec_frozen = self.b_dec.detach()
            via_gate_reconstruction = (via_gate_feature_magnitudes @ W_dec_frozen + b_dec_frozen)
            L_aux = (x - via_gate_reconstruction).pow(2).sum(-1).mean(0)
            l_aux_time = time.time()

            # Summing up the losses
            loss = L_Reconstruct + L_Sparsity + L_aux
            loss_time = time.time()

            print(f"Total time: {loss_time - start_time:.6f}s")
            print(f"time in with_loss: {loss_time - reconstruct_time:.6f}s")
            return GatedAutoEncoderResult(
                loss=loss, 
                x_reconstruct=x_reconstruct, 
                acts=acts, 
                L_Reconstruct=L_Reconstruct, 
                L_Sparsity=L_Sparsity, 
                L_aux=L_aux
            )
        else:
            print(f"Total time: {reconstruct_time - start_time:.6f}s")
            return x_reconstruct
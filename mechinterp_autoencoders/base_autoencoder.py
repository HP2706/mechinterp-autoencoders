import torch
from typing import Any, Literal, Optional, Self, Union, overload
from pydantic import BaseModel
import os
from abc import ABC, abstractmethod
import inspect
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from jaxtyping import Float

class AutoEncoderBaseConfig(BaseModel):
    '''Base config for all autoencoders'''
    dict_mult: int
    d_input: int 
    seed: int = 42
    enc_dtype: torch.dtype = torch.float32
    device: torch.device = torch.device('cuda')
    updated_anthropic_method : bool = True

    class Config:
        arbitrary_types_allowed = True
    
    @property
    def d_sae(self):
        return self.d_input * self.dict_mult

    @property 
    def dtype(self):
        return torch.float32 if self.enc_dtype == 'fp32' else torch.float16
    
    @property
    def save_name(self) -> str:
        #name to save
        name = self.__class__.__name__
        return f'{name}_d_hidden_{self.d_input * self.dict_mult}_dict_mult_{self.dict_mult}'


class AutoEncoderBase(nn.Module, ABC):
    def __init__(self, cfg : AutoEncoderBaseConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):...

    @property
    def d_sae(self):
        return self.cfg.d_sae
    
    @abstractmethod
    def forward(self, x: Float[Tensor, 'batch d_in']):
        """
        Abstract forward method that must be implemented by subclasses.
        The method parameter can dictate the behavior of the forward pass.
        """
        pass

    @abstractmethod
    def _prepare_params(self, feature_indices: Optional[slice]):
        '''
        this is useful for when you only want to look at a 
        subset of the weights to save compute
        '''
        pass

    @abstractmethod
    def encode(self, x: Float[Tensor, 'batch d_in']) -> Float[Tensor, 'batch d_sae']:
        '''
        Abstract encode method that must be implemented by subclasses.
        '''
        pass

    @abstractmethod
    def decode(self, x: Float[Tensor, 'batch d_sae']) -> Float[Tensor, 'batch d_in']:
        '''
        Abstract decode method that must be implemented by subclasses.
        '''
        pass

    @torch.no_grad()
    def get_weight_data(self) -> dict[str, float]:
        weight_data = {}
        for name, value in inspect.getmembers(self):
            if isinstance(value, Union[nn.Parameter, nn.Linear]):
                weight_data[f'{name}_norm'] = torch.norm(value.data).item()
        return weight_data

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()
    
    @property
    def name(self):
        return self.__class__.__name__.lower()

    @overload
    @classmethod
    def load_from_checkpoint(
        cls, 
        save_path : str, 
    ) -> Self:
        ...

    @overload
    @classmethod
    def load_from_checkpoint(
        cls, 
        save_path : str, 
    ) -> tuple[Self, dict]:
        ...

    @classmethod
    def load_from_checkpoint(
        cls, 
        save_path : str, 
    ) -> Union[tuple[Self, dict], Self]:
        """
        Loads the saved autoencoder from a checkpoint
        
        Args:
            save_path (str): Path to the checkpoint containing the model, train config, and model config

        Returns:
            tuple[AutoEncoderBase, dict]: The autoencoder and the train config
        """
        checkpoint = torch.load(save_path)
        model_config = checkpoint['model_config']
        model = cls(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        train_config = checkpoint.get('train_config', {})
        return model, train_config
    
    def save_model(
        self,
        save_path : str,
        train_cfg: Optional[dict[str, Any]] = None,
    ):
        '''
        Args:
            save_path (str): Path to save the model
            train_cfg (Optional[dict[str, Any]]): optionally save the train config with the model
        '''
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.cfg.model_dump(),
        }
        if train_cfg is not None:
            checkpoint['train_config'] = train_cfg
        torch.save(checkpoint, save_path)
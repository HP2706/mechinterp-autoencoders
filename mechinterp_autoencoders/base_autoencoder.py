import torch
import time
from typing import Any, Literal, Optional, Self, Union, cast, overload
from pydantic import BaseModel
import os
from abc import ABC, abstractmethod
import inspect
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from tqdm import tqdm
from contextlib import contextmanager
from mechinterp_autoencoders.utils import extract_nonzero

if torch.cuda.is_available():
    from .kernels import TritonDecoder

class AutoEncoderBaseConfig(BaseModel):
    '''Base config for all autoencoders'''
    dict_mult: int
    d_input: int 
    seed: int = 42
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tie_w_dec: bool = True #whether to tie encoder decoder weights at initialization
    use_kernel: bool = True #uses top k eager decoding
    use_pre_enc_bias: bool = True #uses a pre-bias for the encoder
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def d_sae(self):
        return self.d_input * self.dict_mult
    
    @property
    def save_name(self) -> str:
        #name to save
        name = self.__class__.__name__
        return f'{name}_d_hidden_{self.d_input * self.dict_mult}_dict_mult_{self.dict_mult}'

class AbstractAutoEncoder(nn.Module, ABC):
    
    @abstractmethod
    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):...
    
    @abstractmethod
    def forward(self, x: Float[Tensor, 'batch d_in'], method: Literal['reconstruct', 'with_loss','acts']):
        """
        Abstract forward method that must be implemented by subclasses.
        The method parameter can dictate the behavior of the forward pass.
        """
        pass

    @abstractmethod
    def _prepare_params(self, x: Tensor, feature_indices: Optional[slice]):
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

class BaseAutoEncoder(AbstractAutoEncoder):
    '''
    this autoencoder is a basic autoencoder that 
    implements certain stable methods that multiple autoencoders will use
    '''
    cfg: AutoEncoderBaseConfig #must inherit from base cfg class
    W_enc: torch.Tensor
    pre_bias: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor

    def initialize_weights(self):
        '''
        a helper method that initializes the following weights
        W_enc : (d_input, d_sae)
        pre_bias : (d_sae)
        W_dec : (d_sae, d_input)
        b_enc : (d_sae)
        '''
        self.W_enc = nn.Parameter(torch.randn(self.cfg.d_input, self.cfg.d_sae, dtype=self.cfg.dtype))
        self.W_dec = nn.Parameter(torch.randn(self.cfg.d_sae, self.cfg.d_input, dtype=self.cfg.dtype).contiguous())

        assert self.W_dec.is_contiguous(), "W_dec must be contiguous"
        self.pre_bias = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=self.cfg.dtype))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype))

        if self.cfg.tie_w_dec:
            self.tie_weights()

    def tie_weights(self):
        '''
        ties the encoder to the transpose of the decoder weights
        '''
        if self.cfg.tie_w_dec:
            self.W_enc.data = self.W_dec.data.clone().T.contiguous()

    @contextmanager
    def _prepare_params(self, x: Tensor, feature_indices: Optional[slice]):
        if feature_indices is None:
            yield x
        else:
            original_W_enc = self.W_enc
            original_pre_bias = self.pre_bias
            original_W_dec = self.W_dec
            
            self.W_enc = self.W_enc[feature_indices, :]
            self.pre_bias = self.pre_bias[feature_indices]
            self.W_dec = self.W_dec[:, feature_indices]
            x = x[:, feature_indices]
            try:
                yield x
            finally:
                self.W_enc = original_W_enc
                self.pre_bias = original_pre_bias
                self.W_dec = original_W_dec

    # inspired by https://github.com/EleutherAI/sae/blob/main/sae/utils.py#L94
    def eager_decode(
        self, 
        top_indices : Int[Tensor, 'batch k'],
        top_acts : Float[Tensor, 'batch k'],
        feature_indices: Optional[slice] = None
    ) -> Tensor:
        '''
        Decode method assumes self.W_dec and self.pre_bias are defined
        '''
        with self._prepare_params(top_acts, feature_indices):
            buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.cfg.d_sae,))
            acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
            assert buf.shape == (top_acts.shape[0], self.cfg.d_sae), f"expected shape (top_acts.shape[0]/batch, self.cfg.d_sae) got {buf.shape}"
            return acts @ self.W_dec
        
    def kernel_decode(
        self, 
        top_indices : Int[Tensor, 'batch k'], 
        top_acts : Float[Tensor, 'batch k'], 
        feature_indices: Optional[slice] = None
    ) -> Float[Tensor, 'batch d_in']:
        '''
        Decode method assumes self.W_dec and self.pre_bias are defined
        '''
        with self._prepare_params(top_acts, feature_indices):
            if torch.cuda.is_available():
                return cast(
                    Tensor,
                    TritonDecoder.apply(top_indices, top_acts, self.W_dec.mT)
                )
            else:
                raise ValueError("Triton decoder is not available on non cuda devices")

    def decode(
        self, 
        acts : Float[Tensor, 'batch d_sae'], 
        feature_indices: Optional[slice] = None
    ) -> Float[Tensor, 'batch d_in']:
        '''
        Decode method assumes self.W_dec and self.pre_bias are defined
        '''
        with self._prepare_params(acts, feature_indices):
            if self.cfg.use_kernel:
                non_zero_values, non_zero_indices = extract_nonzero(acts)
                if torch.cuda.is_available():
                    decoded = self.kernel_decode(non_zero_indices, non_zero_values) 
                else:
                    decoded = self.eager_decode(non_zero_indices, non_zero_values) 
            else:
                decoded = acts @ self.W_dec

            return decoded + self.pre_bias #apply bias
    
    def zero_optim_grads(self, optimizer : Optimizer, indices : torch.Tensor):
        for dict_idx, (k, v) in tqdm(enumerate(optimizer.state.items()), desc="setting gradients to zero"):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.cfg.d_sae, self.cfg.d_input), f"expected shape (self.cfg.d_sae, self.cfg.d_input) got {k.data.shape}"
                    v[v_key][indices, :] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.cfg.d_input, self.cfg.d_sae), f"expected shape (self.cfg.d_input,) got {k.data.shape}"
                    v[v_key][:, indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.cfg.d_sae,), f"expected shape (self.cfg.d_input, self.cfg.d_sae) got {k.data.shape}"
                    v[v_key][indices] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.cfg.d_input,), f"expected shape (self.cfg.d_sae,) got {k.data.shape}"
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
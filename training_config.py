from pydantic import BaseModel, Field
from typing import List, Optional, Literal
#from autoencoder import AUTOENCODER_TYPES
from _types import Loss_Method

class AutoencoderTrainConfig(BaseModel):
    type : Literal['autoencoder', 'gated_autoencoder', 'topk_autoencoder']
    wandb_log : bool = True
    normalize_w_dec : bool = Field(True, description="""
        normalize the wdec to unit norm after each step necessary to 
        avoid gameability of sparsity metric
        """
    )
    batch_size: int
    num_tokens: Optional[int] = None
    l1_coeff: float
    with_ramp : bool = False
    lr : float
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int
    n_epochs: int 
    n_steps: int
    test_steps : int
    training_set : Optional[List[str]] = None
    validation_set : Optional[List[str]] = None
    loss_func : Optional[Loss_Method] = 'with_loss'
    anthropic_resampling : bool = False
    anthropic_resample_look_back_steps : Optional[int] = None
    sched_lr_factor : Optional[float] = None
    save_interval : int = 1
    resampling_interval : Optional[int] = None
    resample_factor : float = 0.2
    bias_resample_factor : float = 0.2
    resampling_dataset_size : int = 1000


    @classmethod
    def dummy_default(cls): # only for testing
        return cls(
            type='autoencoder',
            normalize_w_dec=True,
            n_epochs=1,
            n_steps=1,
            lr=1e-3,
            batch_size=1,
            num_tokens=1,
            l1_coeff=1e-3,
            with_ramp=False,
            loss_func='with_loss',
            anthropic_resampling=False,
            anthropic_resample_look_back_steps=1,
            sched_lr_factor=None,
            test_steps=1,
            training_set=None,
            validation_set=None,
            save_interval=1,
            resampling_interval=None,
            resampling_dataset_size=1,
            bias_resample_factor=0.2,
            resample_factor=0.2
        )
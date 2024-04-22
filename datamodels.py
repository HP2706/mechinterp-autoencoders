from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional
import torch


class RunMetaData(BaseModel):
    n_epoch: int
    loss: float
    is_validation: bool = False
    gpu_usage_percent: float = 0.0

class ActivationData(BaseModel):
    text: Optional[str]
    activations: Any
    token_ids : List[int]

class ActivationMetaData(BaseModel):
    n_saved: int 
    last_idx: int 

class InterpretabilityExample(BaseModel):
    text: str
    token_ids: List[int]
    activations: Any


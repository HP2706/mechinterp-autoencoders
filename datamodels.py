from pydantic import BaseModel
from typing import Any, List, Optional

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


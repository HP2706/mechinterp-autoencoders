from pydantic import BaseModel
from typing import Any

class RunMetaData(BaseModel):
    n_epoch: int
    loss: float
    is_validation: bool = False
    gpu_usage_percent: float = 0.0

class ActivationData(BaseModel):
    text: str
    activations: Any


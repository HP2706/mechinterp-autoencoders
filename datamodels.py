from pydantic import BaseModel

class RunMetaData(BaseModel):
    n_epoch: int
    loss: float
    is_validation: bool = False
    gpu_usage_percent: float = 0.0
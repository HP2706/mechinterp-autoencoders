from pydantic import BaseModel

class TrainMetaData(BaseModel):
    n_epoch: int
    loss: float
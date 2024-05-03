from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional, Literal, Union
import torch

class LaionRowData(BaseModel):
    image_url: str
    caption: str
    quantized_activation: int


class PipelineConfig(BaseModel):
    device: Literal['cuda', 'mps', 'cpu']
    d_type: torch.dtype = torch.float32
    batch_size: int = 512
    seq_len: int = 128

    class Config:
        arbitrary_types_allowed = True

class InterpretabilityData(BaseModel):
    feature_or_neuron : Literal["feature", "neuron"]
    index : int
    llm_explanation : str
    spearman_corr : float
    actual_data : Optional[List[int]] = Field(None, description="""quantized activations""")
    llm_predictions : Optional[List[int]] = Field(None, description="""guess on the quantized activation""")


class PredictActivation(BaseModel):
    value : int = Field(..., description="""
        the predicted activation for the feature or neuron in the specified quantized range"""
    )


class ActivationHypothesis(BaseModel):
    hypothesis : str = Field(..., description="""
        an hypothesis for what the feature or neuron is doing based on when the feature or neuron is active.
        This hypothesis should be based on the examples you are given.
    """)
    attributes: str = Field(..., description="""
        the attributes of the feature or neuron, in what situations is it active?
    """)
    reasoning: str = Field(..., description="""
        the reasoning behind the hypothesis, 
        rely on the negative and positive examples you are given
    """)

class TextContent(BaseModel):
    token: str
    token_id: int
    positions : List[int] = Field(..., description="""
        the positions in the text where the token appears, this can be multiple places
    """)
    text: str

class ImageContent(BaseModel):
    image_url: str
    caption: str

class FeatureSample(BaseModel):
    quantized_activation: int
    activation: float
    content : Union[TextContent, ImageContent]
    
class FeatureDescription(ActivationHypothesis):
    index: int
    feature_or_neuron: Literal["feature", "neuron"]
    high_act_samples: list[FeatureSample]
    low_act_samples: list[FeatureSample]

class PredictNextLogit(BaseModel):
    is_next: bool = Field(..., description="""
        based on earlier history, guess if the next token is the suggested token or not, given the context
    """) 

class ActivationExample(BaseModel):
    feature_or_neuron: Literal["neuron", "feature"] = "feature"
    index: int
    activation: float
    content : Union[TextContent, ImageContent]

class MultiTokenActivationExample(BaseModel):
    feature_or_neuron: Literal["neuron", "feature"] = "feature"
    index: int
    tokens: List[str]
    token_ids: List[int]
    activation: List[float]
    text: str

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


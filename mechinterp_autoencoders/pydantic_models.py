import os
from click import Option
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional, Literal, Protocol, Union, overload, runtime_checkable
from mechinterp_autoencoders.utils import format_image_anthropic, format_image_openai
import torch

class AnthropicResample(BaseModel):
    inactive_features: torch.Tensor = Field(..., description="""
        the indices of all features that are not 
        active over last interval(most often 12500 consecutive steps in antrhopic paper)
    """)
    resample_norm : float = Field(..., description="""
        the norm of the resampled features, 
        that which is hopefully less or around one
    """)

    @property
    def inactive_features_count(self) -> int:
        return int(self.inactive_features.sum().item())
    
    class Config:
        arbitrary_types_allowed = True

def save_html(samples : List['FeatureSample'], filename: str):
    assert all(isinstance(elm, FeatureSample) for elm in samples), "All examples should be FeatureSample"

    sorted_samples = sorted(samples, key=lambda x: x.quantized_activation) # sort by quantized activation
    html = ''.join([elm.generate_html() for elm in sorted_samples])
    with open(filename, 'w') as file:
        file.write(html)

class PipelineConfig(BaseModel):
    device: Literal['cuda', 'mps', 'cpu']
    d_type: torch.dtype = torch.float32
    batch_size: int = 512
    seq_len: int = 128

    class Config:
        arbitrary_types_allowed = True

class InterpretabilityMetaData(BaseModel):
    total_samples: int
    mean_prediction: float
    #mse: float
    spearman_corr : float   
    distribution : dict[Union[int, str], int] = Field(..., description="how many samples are in each quantized range")
    actual_quantized_activations : Optional[List[int]] = Field(None, description="""quantized activations""")
    llm_predicted_quantized_activations : Optional[List[int]] = Field(None, description="""guess on the quantized activation""")

class PredictActivation(BaseModel):
    value : int = Field(..., description="""
        the predicted activation for the feature or neuron in the specified quantized range"""
    )

class ActivationHypothesis(BaseModel):
    attributes: str = Field(..., description="""
        the attributes of the feature or neuron, in what situations is it active?
    """)
    reasoning: str = Field(..., description="""
        the reasoning behind the hypothesis, 
        rely on the negative and positive examples you are given
    """)
    hypothesis : str = Field(..., description="""
        an hypothesis for what the feature or neuron is doing based on when the feature or neuron is active.
        This hypothesis should be based on the examples you are given.
    """)
    conviction : Optional[int] = Field(None, description="""the level of certainty in the hypothesis from 1-5""") 
    
    @field_validator('conviction')
    def check_conviction(cls, v):
        if v is None:
            return v
        if not 1 <= v <= 5:
            raise ValueError("conviction should be between 1 and 5")
        return v
    
    def stringify(self) -> str:
        return f"""
            ActivationHypothesis:
            Hypothesis: {self.hypothesis}
            Attributes: {self.attributes}
            Reasoning: {self.reasoning}
            Conviction: {self.conviction}
        """



@runtime_checkable
class FormatForAPI(Protocol):
    @overload
    def format_for_api(self) -> List[dict]: ...
    @overload
    def format_for_api(self, image_provider : Literal['openai', 'anthropic', 'gemini']) -> List[dict]: ...
    
    def format_for_api(self, image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai') -> List[dict]:
        ...


class InconclusiveHypothesis(ActivationHypothesis):
    reason: str = Field(..., description="""why the hypothesis is inconclusive""")

    def stringify(self) -> str:
        return f"""
            InconclusiveHypothesis:
            reason: {self.reason}
        """

class TextContent(BaseModel):
    token: str
    token_id: int
    positions : List[int] = Field(..., description="""
        the positions in the text where the token appears, this can be multiple places
    """)
    text: str

    def format_for_api(self ) -> List[dict]:
        return [
            { #type: ignore
                "type": "text", 
                "text": f"""
                    text:{self.text}\n
                    token:{self.token} (ID: {self.token_id})\n
                    positions(the indexes in the text where the token appears):{self.positions}
                """
            }
        ]

class ImageContent(BaseModel):
    image_url: str
    caption: str

    def format_for_api(self, image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai') -> List[dict]:
        return [
            { #type: ignore
                "type": "text", 
                "text": f"""
                    caption:{self.caption}\n
                    for below image
                """

            },
            format_image_openai(self.image_url) if image_provider in ['openai', 'gemini'] 
            else format_image_anthropic(self.image_url)
        ]

class FeatureSample(BaseModel):
    quantized_activation: int
    activation: float
    content : Union[TextContent, ImageContent]
    
    class Config:
        arbitrary_types_allowed=True

    def generate_html(self) -> str:
        html_content = f"<p><strong>Activation:</strong> {self.quantized_activation}</p>"
        if isinstance(self.content, TextContent):
            html_content += f"<p><strong>Text:</strong> {self.content.text}</p>"
            html_content += f"<p><strong>Token:</strong> {self.content.token} (ID: {self.content.token_id})</p>"
            html_content += f"<p><strong>Positions:</strong> {self.content.positions}</p>"
        elif isinstance(self.content, ImageContent):
            html_content += f"<img src='{self.content.image_url}' alt='{self.content.caption}' style='width:300px;'><br>"
            html_content += f"<caption>{self.content.caption}</caption>"
        return html_content

    
    def format_for_api(
        self, 
        image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai'
    )-> List[dict]:
        if isinstance(self.content, ImageContent) :
            content = self.content.format_for_api(image_provider)
        else:
            content = self.content.format_for_api()
        return [
            { #type: ignore
                "type": "text", 
                "text": f"""
                    quantized_activation:{self.quantized_activation}
                    for the below data
                """
            },
            *content
        ]

    
class FeatureDescription(BaseModel):
    index: int
    activation_hypothesis: ActivationHypothesis
    feature_or_neuron: Literal["feature", "neuron"]
    high_act_samples: list[FeatureSample]
    low_act_samples: list[FeatureSample]
    metadata: Optional[InterpretabilityMetaData] = None
    used_indices: List[int]
    total_feature_distribution: Optional[dict[int, int]] = Field(None, description="how many samples are in each quantized range")

    def set_metadata(self, metadata: InterpretabilityMetaData):
        self.metadata = metadata

    def display_in_file(self, folder_path: str):
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Create a unique filename for this feature
        filename = f"feature_{self.index}_{self.feature_or_neuron}.html"
        file_path = os.path.join(folder_path, filename)

        # Start HTML document
        html_content = f"<html><head><title>Feature Description for {self.feature_or_neuron} {self.index}</title></head><body>"
        html_content += f"<h1>Hypothesis: {self.activation_hypothesis.hypothesis}</h1>"
        html_content += f"<h2>Attributes: {self.activation_hypothesis.attributes}</h2>"
        html_content += f"<h3>Reasoning: {self.activation_hypothesis.reasoning}</h3>"

        # Add high activation samples
        html_content += "<h2>High Activation Samples:</h2>"
        for sample in self.high_act_samples:
            html_content += sample.generate_html()

        # Add low activation samples
        html_content += "<h2>Low Activation Samples:</h2>"
        for sample in self.low_act_samples:
            html_content += sample.generate_html()
        # Close HTML document
        html_content += "</body></html>"

        # Write to file
        with open(file_path, "w") as file:
            file.write(html_content)

class PredictNextLogit(BaseModel):
    is_next: bool = Field(..., description="""
        based on earlier history, guess if the next token is the suggested token or not, given the context
    """) 

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


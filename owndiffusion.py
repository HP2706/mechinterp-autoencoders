import modal
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers import StableUnCLIPImg2ImgPipeline
from transformers import CLIPProcessor, CLIPModel

class AutoEncoderPipeLine:
    def __init__(
        self, 
        tokenizer: CLIPModel, 
        text_model: CLIPModel,
        model_id: str = "stabilityai/stable-diffusion-2-1-unclip-small",
        data_type: torch.dtype = torch.float16,
    ):
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=data_type,
            prior_tokenizer=tokenizer,
            prior_text_encoder=text_model,
        )

    @staticmethod
    def from_pretrained(
        diffusion_model_id: str,
        clip_model_id: str = "openai/clip-vit-large-patch14", 
    ):
        tokenizer = CLIPModel.from_pretrained(clip_model_id)
        text_model = CLIPModel.from_pretrained(clip_model_id)
        return AutoEncoderPipeLine(tokenizer, text_model, diffusion_model_id) # type: ignore

    def gen_img(self, prompt: str, image_embedding: torch.Tensor):
        return self.pipe( # type: ignore
            prompt=prompt, image_embeds=image_embedding.to("cuda").to(torch.float16) # type: ignore
        ).images

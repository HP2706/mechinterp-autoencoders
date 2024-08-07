import functools
import time
from typing import Optional, Union
from PIL import Image
import io
import base64
import beartype
import requests
import torch
from jaxtyping import Float, jaxtyped, Int
from beartype import beartype
from torch import Tensor

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def generate_sparse_tensor(
    size :tuple[int, int],
    sparsity: float,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    n_elements = size[0] * size[1]
    n_nonzero = int(n_elements * sparsity)
    x = torch.zeros(size, device=device, dtype=dtype)
    indices = torch.randperm(n_elements, device=device)[:n_nonzero]
    # Fill the selected indices with random non-zero values
    x.view(-1)[indices] = torch.randn(n_nonzero, device=device, dtype=dtype)
    return x

def extract_nonzero(
    x : Float[Tensor, "batch_size d_sae"],
    k : Optional[int] = None
) -> tuple[
    Float[Tensor, "batch_size a"], 
    Int[Tensor, "batch_size a"]
]:
    '''
    Args:
        x: Tensor of shape (batch_size, d_sae)
        k: Number of non-zero elements to extract
    Returns:
        top_vals: Tensor of shape (batch_size, a)
        top_indices: Tensor of shape (batch_size, a)
    
    if k is not none topk is used, else, k=maximum number of non zeros per batch.
    this massively speeds up inference 
    '''
    # Find the max number of non-zero elements in the batch
    if k:
        #this is a lot faster when k is know in advance
        top_vals, top_indices = torch.topk(x, k=k, dim=-1)
        return top_vals.contiguous(), top_indices.contiguous()
    else:
        max_non_zero_elms = max(1, int((x != 0).sum(dim=-1).max()))
        _, top_indices = x.abs().topk(max_non_zero_elms, sorted=False)
        top_acts = x.gather(dim=-1, index=top_indices)
        return top_acts.contiguous(), top_indices.contiguous()



def format_image_anthropic(img: Union[Image.Image, str]) -> dict:
    if isinstance(img, str):
        img = Image.open(io.BytesIO(requests.get(img).content))
        
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_media_type = "image/jpeg"

    return {
        "type": "image", 
        "source": {
            "type": "base64",
            "media_type": image_media_type,
            "data": base64_image
        }
    }

def format_image_openai(img: Union[Image.Image, str]) -> dict:
    if isinstance(img, str):
        return {
            "type": "image_url",
            "image_url": {
                "url": img
            }
        }
    else:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

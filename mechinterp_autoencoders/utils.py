import functools
import time
from typing import Union
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

@jaxtyped(typechecker=beartype)
def extract_nonzero(
    x : Float[Tensor, "batch_size d_sae"]
) -> tuple[
    Float[Tensor, "batch_size a"], 
    Int[Tensor, "batch_size a"]
]:
    '''
    takes d_sae dimensional batch and return a tensor that has as many dimensions as the maximum non_zero elements in the batch
    this is for instance useful at inference time when the tensors are extremely sparse or late in training
    '''
    # Find the max number of non-zero elements in the batch
    max_non_zero_elms = int((x.abs() > 1e-5).sum(dim=-1).max())
    topk = torch.topk(x.abs(), k=max_non_zero_elms, dim=-1)
    sorted_values = torch.gather(x, -1, topk.indices)
    return sorted_values.contiguous(), topk.indices.contiguous()



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

def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken for {func.__name__} is {time.time() - start}")
        return result
    return wrapper
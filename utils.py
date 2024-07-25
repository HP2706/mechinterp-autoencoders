import functools
import time
from typing import Union
from PIL import Image
import io
import base64
import requests
import torch

def extract_nonzero(x):
    batch_size, seq_len = x.shape
    mask = (x.abs() > 1e-5).long()
    a = mask.sum(1).max().item()

    values = []
    indices = []
    for i in range(batch_size):
        non_zero_idxs = torch.nonzero(mask[i]).squeeze()
        num_non_zero = non_zero_idxs.numel()
        
        if non_zero_idxs.dim() == 0:
            non_zero_idxs = non_zero_idxs.unsqueeze(0)
        num_non_zero = non_zero_idxs.numel()
        # Get all possible indices
        all_indices = torch.arange(seq_len, device=x.device)
        zero_idxs = all_indices[~torch.isin(all_indices, non_zero_idxs)]
        # Randomly select padding indices
        padding_idxs = zero_idxs[torch.randperm(zero_idxs.numel())[:a - num_non_zero]]
        
        # Combine non-zero and padding indices
        row_indices = torch.cat([non_zero_idxs, padding_idxs]) if non_zero_idxs.numel() > 0 else padding_idxs
        non_zero_row = x[i,row_indices]
        values.append(non_zero_row)
        indices.append(row_indices)
        
    values = torch.stack(values)
    indices = torch.stack(indices)
    
    assert values.shape == (batch_size, a)
    assert indices.shape == (batch_size, a)
    
    return values, indices

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
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

def generate_sparse_tensor(
    batch_size : int, 
    feature_dim : int, 
    sparsity : float
):
    # Calculate the number of non-zero elements
    n_nonzero = int(batch_size * feature_dim * sparsity)
    x = torch.zeros(batch_size, feature_dim)
    indices = torch.randint(0, batch_size * feature_dim, (n_nonzero,))
    x.view(-1)[indices] = torch.randn(n_nonzero)
    return x


@jaxtyped(typechecker=beartype)
def extract_nonzero(
    x: Float[Tensor, "batch_size seq_len"]
) -> tuple[Float[Tensor, "batch_size a"], Int[Tensor, "batch_size a"]]:
    batch_size, seq_len = x.shape
    mask = (x.abs() > 1e-5).long()
    a = int(mask.sum(1).max().item())
    
    torch.zeros(batch_size, a, device=x.device)
    non_zero_idxs = torch.nonzero(mask)
    row_counts = torch.bincount(non_zero_idxs[:, 0], minlength=batch_size)
    padding_count = a - row_counts

    # Correct way to get zero indices
    values = torch.zeros(batch_size, a, device=x.device, dtype=x.dtype)
    indices = torch.zeros(batch_size, a, device=x.device, dtype=torch.long)

    # crazy vectorized way to get the indices
    # what are you not willing to do to avoid a for loop in python:)
    free_mask = (mask == 0) & (torch.arange(seq_len, device=x.device).unsqueeze(0) < padding_count.unsqueeze(1))
    non_zero_mask = mask.bool()

    indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    indices = torch.where(
        free_mask | non_zero_mask, indices, 
        torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.long)
    )
    indices = indices[:, :a]  #shape (batch_size, a)
    values = x.gather(1, indices)

    return values, indices


def extract_nonzero_for_loop(x):
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
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Optional, Dict, Any, TypeVar, Union, Literal, cast
from typing import cast
from pydantic import BaseModel
import json
from typing import Type, Tuple
import time
import functools
import base64
from PIL import Image
import io
import requests

T = TypeVar("T")
B = TypeVar('B')

def flatten_lst(lst: List[List[T]]) -> List[T]:
    return [item for sublist in lst for item in sublist]

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


import asyncio
import aiohttp

def filter_pairs_by_first(lst : List[Optional[T]], lst2: List[B]) -> Tuple[List[T], List[B]]:
    '''filters both list by None indices in the first list'''
    indices = [i for i, item in enumerate(lst) if item is not None]
    return cast(List[T], [lst[i] for i in indices]),  [lst2[i] for i in indices]

async def async_filter_valid_image_urls(urls: List[str], max_concurrent_requests: int = 50) -> List[bool]:
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def fetch_status(url):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                attempts = 3  # Number of retries
                for _ in range(attempts):
                    try:
                        async with session.head(url, timeout=20) as response:  # Increased timeout
                            if response.status == 200:
                                return True
                    except aiohttp.ClientError:
                        continue
                    await asyncio.sleep(0.1)  # Wait a bit before retrying
                return False

    async def fetch_all(urls):
        return await asyncio.gather(*(fetch_status(url) for url in urls))
    return await fetch_all(urls)

def filter_valid_image_urls(urls : List[str]) -> List[bool]:
    valid_data = []
    for item in urls:
        try:
            response = requests.head(item, timeout=5)
            if response.status_code == 200:
                valid_data.append(True)
            else:
                valid_data.append(False)
        except requests.RequestException:
            # Handle exceptions for timeouts, connection problems, etc.
            valid_data.append(False)
    return valid_data

def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken for {func.__name__} is {time.time() - start}")
        return result
    return wrapper

def get_model_memory_usage(numbers, dtype) -> float:
    '''returns memory in GB for n numbers of dtype'''
    print(f"Memory usage for {numbers} {dtype} numbers")
    memory_bytes = numbers * torch.finfo(dtype).bits // 8
    return memory_bytes / 1024**2

def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

def modified_lm_cross_entropy_loss(logits, tokens):
    loss_fn = CrossEntropyLoss()
    logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    tokens = tokens[:, 1:].contiguous().view(-1)
    return loss_fn(logits, tokens)

def get_device() -> Literal['cuda', 'cpu', 'mps']:
    '''returns the device being used'''
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available(): #type: ignore
        return 'mps'
    else:
        return 'cpu'

def get_gpu_memory_usage() -> float:
    '''returns the percentage of GPU memory used'''
    return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

def load_activations(df, batch_size)-> List[torch.Tensor]:
    loaded_array = np.array(list(df['activations']), dtype=np.float32)
    elms = []
    for i in range(0, len(loaded_array), batch_size):
        elms.append(torch.tensor(loaded_array[i:i+batch_size]))

    sum_batches = sum([len(elm) for elm in elms])
    assert len(df) == sum_batches, f"expected {len(elms)} should be equal to sum_batches: {sum_batches}"
    return elms

def find_token_pos(
    token : int, 
    tokens : torch.Tensor
) -> List[int]:
    idxs = torch.nonzero(tokens == token).squeeze().tolist() 
    return [idxs] if isinstance(idxs, int) else idxs

def load_tensor(
    path : str,
) -> torch.Tensor:
    return torch.tensor(np.load(path))


def get_sparsity_factor(x : torch.Tensor):
    '''returns the sparsity factor of a tensor'''
    mask = x == 0
    return mask.float().sum() / x.numel()


def filter_non_zero_batch(
    activations: torch.Tensor,
    threshold: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
   
    if threshold:
        mask = torch.any(torch.abs(activations) > threshold, dim=-1)
    else:
        mask = torch.any(activations != 0, dim=-1)

    non_zero_indices = mask.nonzero(as_tuple=False)
    zero_indices = (~mask).nonzero(as_tuple=False)
    return non_zero_indices.reshape(-1), zero_indices.reshape(-1)

T = TypeVar("T")

def remove_keys(d : Dict[T, Any], key : Union[List[T], T]):
    if isinstance(key, list):
        for k in key:
            d.pop(k, None)
    else:
        d.pop(key, None)
    return d


C = TypeVar("C", bound=BaseModel)

def convert_to_pydantic_model(target : Type[C], data : dict) -> C:
    '''converts a dictionary to a pydantic model, removes the keys that are not in the model already'''
    model_keys = target.model_fields.keys()
    filtered_data = {k: v for k, v in data.items() if k in model_keys}
    return target(**filtered_data)

B = TypeVar('B', bound=BaseModel)

def write_to_json(data: Dict[int, B], filename: str) -> None:
    '''Writes a list of BaseModel derived objects to a JSON file.'''
    with open(filename, 'w') as file:
        json.dump({k: v.model_dump() for k, v in data.items()}, file, indent=4)

def clip_embed_image(images: List[Image.Image]):
    '''this function embeds images using the CLIP model from OpenAI.'''
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True) # type: ignore
    outputs = model(**inputs) # type: ignore
    return outputs.image_embeds



def test_loss_fn():
    # Test the functions
    batch_size = 2
    sequence_length = 5
    vocab_size = 10

    # Create random logits and tokens
    logits = torch.randn(batch_size, sequence_length, vocab_size)
    tokens = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Ensure tokens for modified loss are shifted correctly
    tokens_for_modified_loss = torch.cat((tokens[:, 1:], torch.zeros(batch_size, 1).long()), dim=1)

    # Calculate losses
    original_loss = lm_cross_entropy_loss(logits, tokens)
    modified_loss = modified_lm_cross_entropy_loss(logits, tokens_for_modified_loss)

    print("Original Loss:", original_loss.item())
    print("Modified Loss:", modified_loss.item())
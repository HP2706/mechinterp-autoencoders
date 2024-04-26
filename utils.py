import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Optional, Dict, Any, TypeVar, Union
from pydantic import BaseModel
import json
from typing import Type


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

def filter_non_zero_sequence(   
    tokens : torch.Tensor,
    activations: torch.Tensor,
    threshold : Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    '''filters out the non-zero activations
    activations shape is (batch_size, seq_len) or (seq_len)
    Args:
        tokens: (batch_size, seq_len) or (seq_len)
        activations: (batch_size, seq_len) or (seq_len)
    Returns:
        non_zero_activations : (n_non_zero_activations)
        non_zero_tokens : (n_non_zero_activations)
    '''
    non_zero_indices = torch.nonzero(activations, as_tuple=True)
    non_zero_activations = activations[non_zero_indices]
    non_zero_tokens = tokens[non_zero_indices]
    if threshold is not None:
        condition = non_zero_activations > threshold
        non_zero_activations = non_zero_activations[condition]
        non_zero_tokens = non_zero_tokens[condition]
    return non_zero_activations, non_zero_tokens

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

def write_models_to_json(models: List[B], filename: str) -> None:
    '''Writes a list of BaseModel derived objects to a JSON file.'''
    with open(filename, 'w') as file:
        json_data = [model.model_dump() for model in models]
        json.dump(json_data, file, indent=4)

def load_models_from_json(model_class: Type[B], filename: str) -> List[B]:
    '''Loads a list of BaseModel derived objects from a JSON file.'''
    with open(filename, 'r') as file:
        json_data = json.load(file)
        models = [model_class(**data) for data in json_data]
    return models

def filter_zeros(
    a : torch.Tensor, 
    threshold : float = 1e-4
):
    '''filters across batch dimension'''
    # Step 1: Identify batches where all elements are zero in the sequence dimension
    zero_batches = torch.all(torch.abs(a) < threshold, dim=1)

    # Step 2: Filter out batches that are entirely zero
    filtered_a = a[~zero_batches]
    return filtered_a

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
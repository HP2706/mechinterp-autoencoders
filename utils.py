import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List


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
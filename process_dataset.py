from datasets import load_dataset
from models import Transformer, TransformerConfig
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from datamodels import RunMetaData
from typing import List, Tuple
from utils import get_model_memory_usage, lm_cross_entropy_loss, get_gpu_memory_usage
import torch
from common import stub, PATH, vol, DATASET_NAME, image
from modal import Image, Volume, gpu

with image.imports():
    import os
    import torch
    from datasets import load_dataset, Dataset, load_from_disk
    from multiprocessing import Pool
    from tqdm import tqdm
    from functools import partial

def process_batch(batch, seq_len : int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer.batch_encode_plus( # type: ignore
        batch, 
        add_special_tokens=True, 
        max_length=seq_len, 
        padding="max_length", 
        truncation=True
    )

def batch_tokenize(data) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    batch_size = 512
    tokens = []
    attn_masks = []
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    process_fn_partial = partial(process_batch, seq_len=512)
    with Pool(processes=os.cpu_count()) as pool:
        for result in tqdm(pool.imap(process_fn_partial, batches), total=len(batches)):
            if isinstance(result, Exception):
                raise result
            tokens.extend(result['input_ids']) # type: ignore
            attn_masks.extend(result['attention_mask']) # type: ignore
    return tokens, attn_masks

@stub.function(
    image = image, 
    volumes={PATH: vol},       
    timeout=30*60, #30 minutes
    cpu=10, #10 cores
    memory=20*1000 #20 GB
)
def download_and_tokenize_dataset():
    model_name = "thesephist/contra-bottleneck-t5-small-wikipedia"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if not os.path.exists(f"{PATH}/original_{DATASET_NAME}"):
        print("Downloading original dataset")
        os.makedirs(f"{PATH}/original_{DATASET_NAME}")
        dataset = load_dataset(DATASET_NAME, num_proc=os.cpu_count())
        dataset.save_to_disk(f"{PATH}/original_{DATASET_NAME}") # type: ignore
        vol.commit()
    else:
        dataset = load_from_disk(f"{PATH}/original_{DATASET_NAME}")
    
    os.makedirs(f"{PATH}/processed_{DATASET_NAME}", exist_ok=True)
        

    for split in dataset.keys():
        inner_dataset : pd.DataFrame = dataset[split].to_pandas() # type: ignore
        # Apply tokenization in parallel
        token_ids, attn_masks = batch_tokenize(inner_dataset['text'].tolist())
        inner_dataset['token_ids'] = token_ids
        inner_dataset['attn_masks'] = attn_masks

        path = f"{PATH}/processed_{DATASET_NAME}_{split}"
        os.makedirs(path, exist_ok=True)
        Dataset.from_pandas(inner_dataset, split=split).save_to_disk(path)
        vol.commit()

        print(f"Saved {split} dataset to {path}")
        print(f"check loading", load_from_disk(path).to_pandas().head(5)) # type: ignore



from datasets import load_dataset
from models import Transformer, TransformerConfig
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from datamodels import RunMetaData
from typing import List
from utils import get_model_memory_usage, lm_cross_entropy_loss, get_gpu_memory_usage
import torch
from common import stub, PATH, vol
from modal import Image, Volume, gpu
import modal
import wandb 
import json
import re
DATASET_NAME = "roneneldan/TinyStories"

image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.11"
).pip_install(
    "torch",
    "transformers",
    "datasets",
    "einops",
    "pandas",
    "pydantic>=2.0",
    "wandb",
    "tqdm",
)

with image.imports():
    import os
    import torch
    from datasets import load_dataset, Dataset, load_from_disk
    from multiprocessing import Pool
    from tqdm import tqdm
    from functools import partial

def process_batch(batch, seq_len : int):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer.batch_encode_plus(
        batch, 
        add_special_tokens=True, 
        max_length=seq_len, 
        padding="max_length", 
        truncation=True
    ).input_ids

def batch_tokenize(data):
    batch_size = 512
    results = []
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    process_fn_partial = partial(process_batch, seq_len=512)
    with Pool(processes=os.cpu_count()) as pool:
        i = 0
        for result in tqdm(pool.imap(process_fn_partial, batches), total=len(batches)):
            if i == 256:
                print("check if max length enforced", torch.tensor(result).shape)
            i += 1
            results.extend(result)
    return results

@stub.function(
    image = image, 
    volumes={PATH: vol},       
    timeout=10*60, #5 minutes
    cpu=10 #10 cores
)
def download_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if not os.path.exists(f"{PATH}/{DATASET_NAME}"):
        os.makedirs(f"{PATH}/{DATASET_NAME}")
        dataset = load_dataset(DATASET_NAME, num_proc=os.cpu_count())
        dataset.save_to_disk(f"{PATH}/{DATASET_NAME}") # type: ignore
        vol.commit()
    else:
        dataset = load_from_disk(f"{PATH}/{DATASET_NAME}")

    for split in dataset.keys():
        inner_dataset = dataset[split].to_pandas() # type: ignore
        print(inner_dataset.head())

        # Apply tokenization in parallel
        inner_dataset['tokenized_text'] = batch_tokenize(inner_dataset['text'].tolist())

        print(inner_dataset.head(1))
        path = f"{PATH}/{DATASET_NAME}/{split}"
        os.makedirs(path, exist_ok=True)
        Dataset.from_pandas(inner_dataset, split=split).save_to_disk(path)
    vol.commit()

#import wandb 
def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

@stub.function(
    image = image,
    volumes={PATH: vol},
    gpu=gpu.A100(memory=40),
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    timeout=120*60, #60 minutes
)
def train_model():
    DATASET_PATH = f"{PATH}/{DATASET_NAME}"
    # start a new wandb run to track this script
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    #first 80% of the dataset is the training set
    
    train_set : Dataset = load_from_disk(f"{DATASET_PATH}/train", keep_in_memory=False) # type: ignore
    val_set : Dataset = load_from_disk(f"{DATASET_PATH}/validation", keep_in_memory=False) # type: ignore
    
    epochs = [1] #, 4, 8, 10]
    for epoch in epochs:
        train_model_epoch(train_set, val_set, epoch) 
    
def train_model_epoch(train_set : Dataset, val_set:Dataset,  epoch : int):
    train_metadata : List[RunMetaData] = []
    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("vocab size before", tokenizer.vocab_size)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("vocab size after", tokenizer.vocab_size)
    cfg = TransformerConfig(
        d_model=64,
        d_vocab=tokenizer.vocab_size +1, #add 1 for the pad token
        init_range=0.02,
        debug=False,
        layer_norm_eps=1e-6,
        n_ctx=512,
        d_mlp=128,
        n_epochs=1,
        batch_size=256,
        lr=1e-3,
        device="cuda",
        beta1=0.9,
        beta2=0.999,
        n_layers=1,
        n_heads=1,
        d_head=64,
        d_type="float32",
        tokenizer_name=tokenizer_name
    )

    transformer_1l = Transformer(cfg)
   
    wandb.init(
        # set the wandb project where this run will be logged
        project="Sparse AutoEncoder",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.lr,
            "architecture": "1Layer Transformer",
            "dataset": "TinyStories",
            "epochs": cfg.n_epochs,
        }
    )

    dir = f'{PATH}/models'
    files = [f for f in os.listdir(dir) if f.endswith(".pth")]
    # Extract numbers from filenames using regex, handle None case, and find the maximum
    n = max([int(re.search(r"num_(\d+)", f).group(1)) for f in files if re.search(r"num_(\d+)", f)] or [0]) + 1 # type: ignore
    runs_dir = f"{dir}/run_{n}"
    os.makedirs(runs_dir, exist_ok=True)

    transformer_1l.to(cfg.device)
    transformer_1l.train()
    print("transformer_1l param count", sum(p.numel() for p in transformer_1l.parameters() if p.requires_grad))
    out = get_model_memory_usage(sum(p.numel() for p in transformer_1l.parameters() if p.requires_grad), torch.float32)
    print("transformer_1l memory usage in mb", out)
    optimizer = AdamW(transformer_1l.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    n_tokens = 0
    for epoch in range(cfg.n_epochs):
        for i in tqdm(range(0, len(train_set), cfg.batch_size)):
            batch = train_set[i:i+cfg.batch_size]['tokenized_text']
            tokens = torch.tensor(batch)
            n_tokens += tokens.numel()
            token_ids = tokens.to(cfg.device)
            print("batch numbers", token_ids.numel())
            mem_usage_float = get_model_memory_usage(token_ids.numel(), torch.float32)
            print("memory usage from batch in mb in float", mem_usage_float)
            print("total memory pressure in gb ", torch.cuda.memory_allocated() / 1024**2)
            (logits, _ ) = transformer_1l.forward(token_ids, return_hidden=False)
            loss = lm_cross_entropy_loss(logits, token_ids) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            data = RunMetaData(n_epoch=epoch, loss=loss_val, gpu_usage_percent=get_gpu_memory_usage()) # this might slow down training meaningfully
            train_metadata.append(data)
            print(data.model_dump())
            wandb.log(data.model_dump())
        print("n_tokens", n_tokens)

    #we test the model is reasonable
    text = "hello who are"
    tokens = tokenizer.encode(text, return_tensors="pt").to(cfg.device) #type: ignore
    print("\n\ntokens", tokens)
    out_tokens = transformer_1l.generate(tokens, 5) #type: ignore
    print("the model response", tokenizer.decode(out_tokens[0]))

    validation_metadata : List[RunMetaData] = []
    #validation 
    transformer_1l.eval()
    for i in tqdm(range(0, len(val_set), cfg.batch_size)):
        batch = val_set[i:i+cfg.batch_size]['tokenized_text']
       
        tokens = torch.tensor(batch).to(cfg.device)
        (logits, _ ) = transformer_1l.forward(tokens, return_hidden=False)
        loss = lm_cross_entropy_loss(logits, tokens)
        loss_val = loss.item()

        data = RunMetaData(
            n_epoch=epoch, 
            loss=loss_val, 
            is_validation=True, 
            gpu_usage_percent=get_gpu_memory_usage()
        )
        validation_metadata.append(data)
        print("stats\n", data.model_dump())
        wandb.log(data.model_dump())

    cfg.n_tokens = n_tokens
    torch.save(transformer_1l.state_dict(), f"{runs_dir}/transformer_1l_num_{n}.pth")
    with open(f"{runs_dir}/transformer_1l_num_{n}.json", "w") as f:
        json.dump(cfg.model_dump_json(), f)

    #we save metadata
    df_train_metadata = pd.DataFrame([data.model_dump() for data in train_metadata])
    df_train_metadata.to_parquet(f"{runs_dir}/train_train_metadata_num_{n}.parquet")
    df_val_metadata = pd.DataFrame([data.model_dump() for data in validation_metadata])
    df_val_metadata.to_parquet(f"{runs_dir}/train_val_metadata_num_{n}.parquet")

    wandb.finish()
    vol.commit()



from datasets import load_dataset
from models import Transformer, TransformerConfig
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from datamodels import TrainMetaData
from typing import List
from utils import get_model_memory_usage, modified_lm_cross_entropy_loss
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
    gpu=gpu.A10G(),
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def train_model():
    train_metadata : List[TrainMetaData] = []
    DATASET_PATH = f"{PATH}/{DATASET_NAME}"
    #first 80% of the dataset is the training set
    
    train_set = load_from_disk(f"{DATASET_PATH}/train", keep_in_memory=False) # type: ignore
    val_set = load_from_disk(f"{DATASET_PATH}/validation", keep_in_memory=False) # type: ignore
    
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
        batch_size=64,
        lr=1e-3,
        device="cuda",
        beta1=0.9,
        beta2=0.999,
        n_layers=1,
        n_heads=1,
        d_head=128,
        d_type="float32",
        tokenizer_name=tokenizer_name
    )

    transformer_1l = Transformer(cfg)
     # start a new wandb run to track this script
    wandb.login(key=os.getenv("WANDB_API_KEY"))
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

    runs_dir = f'{PATH}/models'
    os.makedirs(runs_dir, exist_ok=True)

    transformer_1l.to(cfg.device)
    print("transformer_1l param count", sum(p.numel() for p in transformer_1l.parameters() if p.requires_grad))
    out = get_model_memory_usage(sum(p.numel() for p in transformer_1l.parameters() if p.requires_grad), torch.float32)
    print("transformer_1l memory usage in mb", out)
    optimizer = AdamW(transformer_1l.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    n_tokens = 0
    for epoch in range(cfg.n_epochs):
        batch = []
        for elm in train_set:
            batch.append(elm['tokenized_text']) 
            if len(batch) == cfg.batch_size:
                for elm in batch:
                    print("len of elm", len(elm))
                tokens = torch.tensor(batch)
                n_tokens += tokens.numel()
                print("batch mb size", get_model_memory_usage(tokens.numel(), torch.float32))
                tokens_ids = tokens.to(cfg.device)
                (logits, _ ) = transformer_1l.forward(tokens_ids, return_hidden=False)
                print("logits shape", logits.shape)
                print("tokens_ids shape", tokens_ids.shape)
                loss = modified_lm_cross_entropy_loss(logits, tokens_ids) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                data = TrainMetaData(n_epoch=epoch, loss=loss_val)
                train_metadata.append(data)
                print("stats\n", data.model_dump())
                
                batch = [] # reset batch to empty
                wandb.log(data.model_dump())
                break # we break to check
    print("n_tokens", n_tokens)
    #we test the model is reasonable
    text = "hello who are"
    tokens = tokenizer.encode(text, return_tensors="pt").to(cfg.device) #type: ignore
    print("\n\ntokens", tokens)
    out_tokens = transformer_1l.generate(tokens, 5) #type: ignore
    print("the model response", tokenizer.decode(out_tokens[0]))


    #find the number n 
    files = [f for f in os.listdir(runs_dir) if f.endswith(".pth")]
    # Extract numbers from filenames using regex, handle None case, and find the maximum
    n = max([int(re.search(r"num_(\d+)", f).group(1)) for f in files if re.search(r"num_(\d+)", f)] or [0]) + 1
    torch.save(transformer_1l.state_dict(), f"{runs_dir}/transformer_1l_num_{n}.pth")
    with open(f"{runs_dir}/transformer_1l_num_{n}.json", "w") as f:
        json.dump(cfg.model_dump_json(), f)
    vol.commit()

    #At training steps 25,000, 50,000, 75,000 and 100,000, identify which neurons have not fired in any of the previous 12,500 training steps.
    #Compute the loss for the current model on a random subset of 819,200 inputs.
    #Assign each input vector a probability of being picked that is proportional to the square of the autoencoder’s loss on that input.
    #For each dead neuron sample an input according to these probabilities. Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector for the dead autoencoder neuron.
    #For the corresponding encoder vector, renormalize the input vector to equal the average norm of the encoder weights for alive neurons × 0.2. Set the corresponding encoder bias element to zero.
    #Reset the Adam optimizer parameters for every modified weight and bias term.
    #to test it works, test against randomized transformer models
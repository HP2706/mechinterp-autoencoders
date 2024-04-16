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

DATASET_NAME = "NeelNanda/pile-tokenized-10b"

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
)

@stub.function(
    image = image, 
    volumes={PATH: vol},       
)
def download_dataset():
    from datasets import load_dataset
    import os 
    dataset = load_dataset(DATASET_NAME, num_proc=os.cpu_count())
    dataset.save_to_disk(PATH)
    vol.commit()

#import wandb 
def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

@stub.function(
    image = image,
    volumes={PATH: vol},
    gpu=gpu.A10G()
)
def train_model():
    train_metadata : List[TrainMetaData] = []
    train_set : pd.DataFrame = load_dataset(DATASET_NAME)['train'] # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("vocab size before", tokenizer.vocab_size)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("vocab size after", tokenizer.vocab_size)
    cfg = TransformerConfig(
        d_model=128,
        d_vocab=tokenizer.vocab_size +1, #add 1 for the pad token
        init_range=0.02,
        debug=True,
        layer_norm_eps=1e-6,
        n_ctx=256,
        d_mlp=128,
        n_epochs=10,
        batch_size=128,
        lr=1e-3,
        device="mps",
        beta1=0.9,
        beta2=0.999,
        n_layers=1,
        n_heads=1,
        d_head=128,
        d_type="float32"
    )

     # start a new wandb run to track this script
    """ wandb.init(
        # set the wandb project where this run will be logged
        project="Sparse AutoEncoder",
 
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.lr,
            "architecture": "1Layer Transformer",
            "dataset": "TinyStories",
            "epochs": cfg.n_epochs,
        }
    ) """

    transformer_1l = Transformer(cfg)
    transformer_1l.to(cfg.device)
    print("transformer_1l param count", sum(p.numel() for p in transformer_1l.parameters() if p.requires_grad))
    out = get_model_memory_usage(sum(p.numel() for p in transformer_1l.parameters() if p.requires_grad), torch.float32)
    print("transformer_1l memory usage in mb", out)
    optimizer = AdamW(transformer_1l.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    for epoch in range(cfg.n_epochs):
        for i in range(0, len(train_set), cfg.batch_size):
            batch = train_set[i:i+cfg.batch_size]
            print("batch length", len(batch['text']))
            tokens = tokenizer(list(batch['text']), return_tensors='pt', padding=True, truncation=True, max_length=cfg.n_ctx)
            print("batch mb size", get_model_memory_usage(tokens.input_ids.numel(), torch.float32))
            tokens_ids = tokens.input_ids.to(cfg.device)
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
            print(data.model_dump())
            #wandb.log(data.model_dump())

    #At training steps 25,000, 50,000, 75,000 and 100,000, identify which neurons have not fired in any of the previous 12,500 training steps.
    #Compute the loss for the current model on a random subset of 819,200 inputs.
    #Assign each input vector a probability of being picked that is proportional to the square of the autoencoder’s loss on that input.
    #For each dead neuron sample an input according to these probabilities. Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector for the dead autoencoder neuron.
    #For the corresponding encoder vector, renormalize the input vector to equal the average norm of the encoder weights for alive neurons × 0.2. Set the corresponding encoder bias element to zero.
    #Reset the Adam optimizer parameters for every modified weight and bias term.


#to test it works, test against randomized transformer models
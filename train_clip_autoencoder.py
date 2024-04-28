from common import (
    stub, 
    vol, 
    image, 
    MODELS_DIR,
    PATH, 
    LAION_DATASET_PATH,
    dataset_vol
)
from Laion_Processing.dataloader import LaionDataset, LaionFileLoader

import wandb
from autoencoder import AutoencoderConfig, AutoEncoder, GatedAutoEncoder
from utils import load_activations
from modal import gpu
import modal
from datamodels import ActivationData, ActivationMetaData
import pandas as pd
import os
import numpy as np
import torch
from typing import Optional

with image.imports():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_from_disk
    from huggingface_hub import snapshot_download
    import torch
    from torch.optim import AdamW
    from typing import Any, Dict, List, Type, Union, Generator
    import wandb
    from tqdm import tqdm
    import psutil
    import time
    import os
    import json

@stub.function(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=60*60, 
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def train_autoencoder():

    model_dir = f"{PATH}/1L_autoencoder"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    d_mlp = 768 
    cfg = AutoencoderConfig(
        seed=42,
        batch_size=512*10,
        buffer_mult=10,
        lr=1e-3,
        l1_coeff=0.01,
        beta1=0.9,
        beta2=0.999,
        dict_mult=4,
        seq_len=512,
        d_mlp=d_mlp,
        buffer_size=1000000,
        buffer_batches=25,
        device="cuda",
        n_epochs=10
    )

    model = AutoEncoder(cfg)
    wandb.init(
        # set the wandb project where this run will be logged
        project="Sparse AutoEncoder",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.lr,
            "architecture": "1Layer Transformer",
            "dataset": "Laion2B",
            "epochs": cfg.n_epochs,
        }
    )

    model.to(model.cfg.device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    dataloader = LaionFileLoader(
        batch_size=cfg.batch_size, 
        embeddings_path=f"{LAION_DATASET_PATH}/img_emb"
    ).dataloader
    for epoch in range(model.cfg.n_epochs): # type: ignore
        model.train()
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})
        torch.save(model.state_dict(), f"{model_dir}/laion_model_{epoch}.pt")


        
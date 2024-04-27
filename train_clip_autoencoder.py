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
    
    d_mlp = 512 #TODO this should not be hardcoded
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
            "dataset": "TinyStories",
            "epochs": cfg.n_epochs,
        }
    )

    model.to(model.cfg.device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    for epoch in range(model.cfg.n_epochs): # type: ignore
        for train_file in [ file for file in os.listdir(f"{PATH}/{activations_name}") if "train" in file]:
            if not train_file.endswith(".parquet"):
                continue

            df = pd.read_parquet(f"{PATH}/{activations_name}/{train_file}")
            activations_batched = load_activations(df, model.cfg.batch_size)

            for activations in activations_batched:
                loss, x_reconstruct, acts, l2_loss, l1_loss = model.forward(
                    activations.to(model.cfg.device), method='with_loss'
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"epoch {epoch} train loss {loss.item()}")

        for eval_file in [file for file in os.listdir(f"{PATH}/{activations_name}") if "validation" in file]:
            if not eval_file.endswith(".parquet"):
                continue
            df = pd.read_parquet(f"{PATH}/{activations_name}/{eval_file}")
            activations_batched = load_activations(df, model.cfg.batch_size)
            for activations in activations_batched:
                with torch.no_grad():
                    loss, x_reconstruct, acts, l2_loss, l1_loss = model.forward(
                        activations.to(model.cfg.device), method='with_loss'
                    )
                    print(f"validation loss {loss.item()}")

    save_model(model, model.cfg, model_dir)
    print("model saved")
    print("checking if model can be loaded")
    model, cfg = load_model(model_dir)
    print("model loaded")

        
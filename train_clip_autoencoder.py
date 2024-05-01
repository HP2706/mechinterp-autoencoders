from common import (
    stub, 
    vol, 
    image, 
    METADATA_FOLDER,
    EMB_FOLDER,
    PATH, 
    LAION_DATASET_PATH,
    dataset_vol
)
import aiofiles
import torch
import numpy as np
import asyncio
from io import BytesIO
from Laion_Processing.dataloader import LaionDataset, LaionFileLoader
import wandb
from autoencoder import AutoencoderConfig, AutoEncoder, AutoencoderResult, GatedAutoEncoder, GatedAutoEncoderResult
from utils import load_activations
from modal import gpu
import modal
from datamodels import ActivationData, ActivationMetaData
import pandas as pd
import os
import numpy as np
import torch
from typing import Literal, Optional
from utils import remove_keys

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
    timeout=10*60*60, #10 hours
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True
)
def train_autoencoder(n_epochs: int, type: Literal['autoencoder', 'gated_autoencoder'], dict_mult : int):
    d_mlp = 768 
    paths = [os.path.join(EMB_FOLDER, p) for p in os.listdir(EMB_FOLDER)]
    train_files = paths[:int(len(paths)*0.8)]
    test_files = paths[int(len(paths)*0.8):]

    cfg = AutoencoderConfig(
        seed=42,
        batch_size=2048, #2048 or 4096
        buffer_mult=10,
        lr=5e-5, #anthropic suggested 5e-5
        l1_coeff=0, # initially 0 but progressively increases to 5
        beta1=0.9,
        beta2=0.999,
        dict_mult=dict_mult,
        seq_len=512,
        d_mlp=d_mlp,
        buffer_size=1000000,
        buffer_batches=25,
        device="cuda",
        n_epochs=n_epochs,
        training_set=train_files,
        validation_set=test_files,
        type=type
    )

    model_dir = f"{PATH}/autoencoders"
    os.makedirs(model_dir, exist_ok=True)
    hidden = cfg.d_mlp * cfg.dict_mult
    

    model = AutoEncoder(cfg)
    wandb.init(
        # set the wandb project where this run will be logged
        project="Sparse AutoEncoder",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.lr,
            "architecture": "AutoEncoder",
            "dataset": "Laion2B",
            "epochs": cfg.n_epochs,
        }
    )

    train_loader = LaionFileLoader(
        batch_size=cfg.batch_size, 
        emb_folder=EMB_FOLDER,
        split="train",
        train_share=0.8
    )
    test_loader = LaionFileLoader(
        batch_size=cfg.batch_size, 
        emb_folder=EMB_FOLDER,
        split="test",
        train_share=0.8
    )

    model_path = f"{model_dir}/laion2b_autoencoder"
    os.makedirs(model_path, exist_ok=True)
    vol.commit()

    model.to(model.cfg.device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    step = 0
    total_steps = len(train_loader) * cfg.n_epochs 
    l1_coeff_final = 5
    l1_ramp_steps = int(0.05 * total_steps)
    print("total steps", total_steps*cfg.batch_size)
    print("expected ca total steps", 16*9.5*10e5*cfg.n_epochs)

    
    for epoch in range(cfg.n_epochs): # type: ignore
        model.train()
        for batch in tqdm(train_loader, desc="Training"): # type: ignore
            step += 1
            # Update l1_coeff linearly over the first 5% of the total steps
            if step <= l1_ramp_steps:
                cfg.l1_coeff = (l1_coeff_final / l1_ramp_steps) * step #linearly increase l1_coeff
            else:
                cfg.l1_coeff = l1_coeff_final

            batch = batch.to(model.cfg.device)
            optimizer.zero_grad()
            result = model.forward(batch, method="with_loss")
            result.loss.backward()
            optimizer.step()
            
            data = remove_keys(result.model_dump(), ['x_reconstruct', 'acts'])
            wandb.log({**data, "l1_coeff": cfg.l1_coeff})
            if step % 1000 == 0:
                print({**data, "l1_coeff": cfg.l1_coeff})

        basename = cfg.basename(epoch)
        torch.save(model.state_dict(), f"{model_path}/{basename}.pt")
        with open(f"{model_path}/{basename}.json", "w") as f:
            cfg.n_steps = step
            f.write(cfg.model_dump_json())
        vol.commit()
        print("checkpoint and cfg saved at", f"{model_path}", os.listdir(model_path))

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(model.cfg.device)
                result = model.forward(batch, method="with_loss")

                data = remove_keys(result.model_dump(), ['x_reconstruct', 'acts'])
                wandb.log({**data, "l1_coeff": cfg.l1_coeff})
                print("\n",{**data, "l1_coeff": cfg.l1_coeff})

async def save_activations_async(path: str, activations: torch.Tensor):
    buffer = BytesIO()
    np.save(buffer, activations.numpy(), allow_pickle=False)
    buffer.seek(0)  # Move to the start of the buffer
    async with aiofiles.open(path, 'wb') as f:
        await f.write(buffer.read())
    dataset_vol.commit()

@stub.function(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=60*60, #1 hour
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True,
    gpu=gpu.A10G()
)
def get_recons_loss(
    model_path : str,    
    save_intermediate_acts : bool
):

    autoencoder = AutoEncoder.load_from_checkpoint(
        model_path,
        device="cuda"
    )

    print("len validation set", len(autoencoder.metadata_cfg.validation_set)) # type: ignore
    batch_size = 512*10
    losses = []
    save_tasks = []
    for file in tqdm(autoencoder.metadata_cfg.validation_set):
        try:
            all_tokens = torch.tensor(np.load(file))
        except:
            print("error loading file", file)
            continue
        
        activations = []
        num_batches = len(all_tokens)// (batch_size)
        autoencoder.to(autoencoder.cfg.device)
        autoencoder.eval()
        for batch in tqdm(all_tokens.split(batch_size)[:num_batches]):
            batch = batch.to(autoencoder.cfg.device)
            with torch.no_grad():
                result  = autoencoder.forward(batch, method="with_loss")
            x_reconstruct = result.x_reconstruct
            mean_ablated = batch - batch.mean(dim=0)
            losses.append(((x_reconstruct - mean_ablated)**2).mean().item()) #mse

            if save_intermediate_acts:
                activations.append(result.acts.cpu())
        
        if save_intermediate_acts:
            print("adding activations to queue")
            t0 = time.time()
            activations = torch.vstack(activations)
            filename = file.split('/')[-1].split('.')[0]
            save_path = f"{LAION_DATASET_PATH}/acts_{filename}_{autoencoder.cfg.type}_{autoencoder.file_name}.npy"
            save_tasks.append(save_activations_async(save_path, activations))
            print("time to add to queue", time.time()-t0)
            del activations

    if save_intermediate_acts:
        t0 = time.time()
        print("saving activations to files")
        async def process_and_save_activations(save_tasks):
            await asyncio.gather(*save_tasks)
        # Then replace the existing asyncio.run call with:
        asyncio.run(process_and_save_activations(save_tasks))
        print("time to save all activations to files", time.time()-t0)
        print("chehcking if all files are saved", os.listdir(LAION_DATASET_PATH))
        dataset_vol.commit()
    
    print("mean loss", np.mean(losses))




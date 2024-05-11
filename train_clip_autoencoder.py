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
from torch.nn.utils import clip_grad_norm_
import numpy as np
from io import BytesIO
from Laion_Processing.dataloader import load_loaders
import wandb
from autoencoder import AutoEncoderBase, AutoencoderConfig, AutoEncoder, AutoencoderResult, GatedAutoEncoder, GatedAutoEncoderResult
from utils import load_and_scale_tensor
from modal import gpu
import modal
import os
from typing import Literal, Optional
from utils import remove_keys

from torch.optim import AdamW
from tqdm import tqdm


@stub.function(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=10*60*60, #10 hours
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True
)
def train_autoencoder(
    n_epochs: int, 
    type: Literal['autoencoder', 'gated_autoencoder'], 
    dict_mult : int,
    with_ramp: bool = True,
    loss_func: Literal['with_new_loss', 'with_loss'] = 'with_loss',
    retrain_path : Optional[str] = None,
):
    if retrain_path is not None:
        model = AutoEncoderBase.load_from_checkpoint(retrain_path)
        model_dir = retrain_path
        print("retraining model")
        cfg = model.metadata_cfg
    else:
        d_mlp = 768 
        paths = [os.path.join(EMB_FOLDER, p) for p in os.listdir(EMB_FOLDER)]
        train_files = paths[:int(len(paths)*0.8)]
        test_files = paths[int(len(paths)*0.8):]

        cfg = AutoencoderConfig(
            seed=42,
            batch_size=4096, #2048 or 4096
            buffer_mult=10,
            lr=5e-5, #anthropic suggested 5e-5
            l1_coeff=0, # initially 0 but progressively increases to 5
            beta1=0.9,
            beta2=0.999,
            dict_mult=dict_mult,
            with_ramp=with_ramp,
            loss_func=loss_func,
            seq_len=512,
            d_mlp=d_mlp,
            buffer_size=1000000,
            buffer_batches=25,
            device="cuda",
            n_epochs=0, # inital epochs are 0 but gradually increasing
            training_set=train_files,
            validation_set=test_files,
            type=type
        )

        if type == "gated_autoencoder":
            model = GatedAutoEncoder(cfg)
        else:
            model = AutoEncoder(cfg)


    model_dir = f"{PATH}/laion2b_autoencoders"
    os.makedirs(model_dir, exist_ok=True)
    vol.commit()

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

    train_loader , test_loader = load_loaders(
        batch_size=cfg.batch_size, 
        emb_folder=EMB_FOLDER,
        train_share=0.8,
        n = model.W_dec.shape[0] 
    )

    eighty_percent_steps = len(train_loader) / 0.8
    print("total steps", len(train_loader))
    print("80 percent steps", eighty_percent_steps)


    model.to(model.cfg.device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    step = 0
    
    l1_coeff_final = 5
    total_steps = len(train_loader) * n_epochs 
    #TODO think about effects of retraining here if model.cfg_metadata.l1_coeff is below l1_coeff_final
    l1_ramp = l1_coeff_final / ((total_steps // 100)*5 )
    # we linearly increase l1_coeff from 0 to 5 
    #over the first 5% of the total steps as per anthropic paper

    if retrain_path is not None:
        _range = range(cfg.n_epochs, cfg.n_epochs + n_epochs) 
    else:
        _range = range(n_epochs)

    print("epoch range", _range)
    print("model l1_coeff", model.l1_coeff)
    for epoch in _range: 
        print("epoch", epoch)
        model.train()
        for batch in tqdm(train_loader, total = len(train_loader), desc="dataset training"): 
            step += 1
            # Update l1_coeff linearly over the first 5% of the total steps
            if with_ramp:
                if model.l1_coeff <= l1_coeff_final:
                    model.l1_coeff += l1_ramp 

            batch = batch.to(model.cfg.device)
            #we have scaled the embeddings by 100 to make them larger and easier to work with

            optimizer.zero_grad()
            result = model.forward(batch, method=loss_func) # type: ignore
            result.loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1) 
            # clip grad norm as per anthropic https://transformer-circuits.pub/2024/april-update/index.html#training-saes 
            optimizer.step()
            optimizer.step()
            
            #data = remove_keys(result.model_dump(), ['x_reconstruct', 'acts'])
            if step % 100 == 0:
                wandb.log({
                    "l1_coeff": model.l1_coeff,
                    **result.format_loss()
                })
            if step % 1000 == 0:
                print("step", step)
                print("l1_coeff", model.l1_coeff)
                print("results", result.format_loss())
            
        cfg.n_steps = step
        cfg.l1_coeff = model.l1_coeff
        
        basename = cfg.create_basename()
        model_path = f'{model_dir}/{basename}'
        os.makedirs(model_path, exist_ok=True)
        print("saving model at", model_path)
        torch.save(model.state_dict(), f"{model_path}/model.pt")
        with open(f"{model_path}/config.json", "w") as f:
            f.write(cfg.model_dump_json())
        vol.commit()
        print("checkpoint and cfg saved at", f"{model_dir}", os.listdir(model_dir))

        model.eval()
        with torch.no_grad():
            print(f"evaluation epoch {epoch}\n")
            for batch in tqdm(test_loader[:1000], desc="Testing"): # we only use fraction
                batch = batch.to(model.cfg.device)
                result = model.forward(batch, method="with_loss")
                wandb.log({**result.format_loss(), "l1_coeff": model.l1_coeff})

        cfg.n_epochs += 1


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
):
    autoencoder = AutoEncoder.load_from_checkpoint(
        model_path,
    )

    print("len validation set", len(autoencoder.metadata_cfg.validation_set)) # type: ignore
    batch_size = 512*20
    losses = []
    for file in tqdm(autoencoder.metadata_cfg.validation_set):
        try:
            all_tokens = load_and_scale_tensor(file)
        except:
            print("error loading file", file)
            continue

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
    
    print("mean loss", np.mean(losses))



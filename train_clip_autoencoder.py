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
from utils import load_tensor
from mechninterp_utils import hist
from modal import gpu
import modal
import os
from typing import Literal, Optional, Union

from torch.optim import AdamW
from tqdm import tqdm
from utils import get_device


def save_model(
    model : Union[GatedAutoEncoder, AutoEncoder], 
    cfg: AutoencoderConfig,
    model_dir : str
):
    basename = cfg.create_basename()
    model_path = f'{model_dir}/{basename}'
    os.makedirs(model_path, exist_ok=True)
    print("saving model at", model_path)
    torch.save(model.state_dict(), f"{model_path}/model.pt")
    with open(f"{model_path}/config.json", "w") as f:
        f.write(cfg.model_dump_json())
    vol.commit()
    print("checkpoint and cfg saved at", f"{model_dir}", os.listdir(model_dir))

@stub.function(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=5*60*60, #3 hours
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True
)
def train_autoencoder(
    type: Literal['autoencoder', 'gated_autoencoder'], 
    dict_mult : int,
    steps: int = 10**5, # 100 k steps as per anthropic paper
    save_interval : int = 10**4, # we save the model every save_interval steps
    test_steps : int = 25*10**3,
    with_ramp: bool = True,
    max_l_coef : float = 5,
    loss_func: Literal['with_new_loss', 'with_loss'] = 'with_loss',
    retrain_path : Optional[str] = None,
    resampling_interval : Optional[int] = 10**4,
):
    if with_ramp and loss_func == 'with_loss':
        print("with_ramp does not work for the with_loss loss function disabling it")
        with_ramp = False

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
            lr=0.0012, 
            l1_coeff=0 if with_ramp else 10e-3, 
            beta1=0.9,
            beta2=0.999,
            dict_mult=dict_mult,
            with_ramp=with_ramp,
            loss_func=loss_func,
            seq_len=512,
            d_mlp=d_mlp,
            buffer_size=1000000,
            buffer_batches=25,
            device=get_device(), # type: ignore
            n_epochs=0, # inital epochs are 0 but gradually increasing
            training_set=train_files,
            validation_set=test_files,
            n_steps=0,
            type=type,
            updated_anthropic_method=True if loss_func=='with_new_loss' else False
        )

        if type == "gated_autoencoder":
            model = GatedAutoEncoder(cfg)
        else:
            model = AutoEncoder(cfg)

    def resample_at_step_idx()->bool:
        if resampling_interval is None:
            return False
        else:
            return step % resampling_interval == 0    


    model_dir = f"{PATH}/laion2b_autoencoders/{cfg.folder_name}"
    os.makedirs(model_dir, exist_ok=True)
    vol.commit()


    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(
        # set the wandb project where this run will be logged
        project="Sparse AutoEncoder",
        # track hyperparameters and run metadata
        name = cfg.folder_name,
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
        d_hidden = model.W_dec.shape[0] if cfg.updated_anthropic_method else None,
        n_counts=(None, 1)
    )

    model.to(model.cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    if retrain_path is None:
        step = 0
    else:
        step = model.metadata_cfg.n_steps 
    
    last_resampling_step_idx = step


    #TODO think about effects of retraining here if model.cfg_metadata.l1_coeff is below max_l_coef
    l1_ramp = max_l_coef / ((steps // 100)*5 )
    # we linearly increase l1_coeff from 0 to 5 
    #over the first 5% of the total steps as per anthropic paper

    running_frequency_counter = torch.zeros(model.W_dec.shape[0], dtype=torch.int)

    while step < steps:
        model.train()
        for batch in tqdm(train_loader, total = len(train_loader), desc="dataset training"): 
            step += 1
            # Update l1_coeff linearly over the first 5% of the total steps
            if with_ramp:
                if model.l1_coeff <= max_l_coef:
                    model.l1_coeff += l1_ramp 

            batch = batch.to(model.cfg.device)
            #we have scaled the embeddings by 10 to make them larger and easier to work with

            optimizer.zero_grad()
            result = model.forward(batch, method=loss_func) # type: ignore

            running_frequency_counter += result.did_fire

            result.loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1) 
            # clip grad norm as per anthropic https://transformer-circuits.pub/2024/april-update/index.html#training-saes 
            
            model.remove_parallel_component_of_grads()
            optimizer.step()
            
            metrics = {
                "l1_coeff": model.l1_coeff,
                **result.format_data()
            }

            print("cfg.updated_anthropic_method", cfg.updated_anthropic_method)
            if not cfg.updated_anthropic_method:
                with torch.no_grad():
                    # Renormalize W_dec to have unit norm
                    norms = torch.norm(model.W_dec.data, dim=1, keepdim=True)
                    model.W_dec.data /= (norms + 1e-6) 


            if step % save_interval == 0:
                print("saving model at step", step)
                save_model(model, cfg, model_dir)
            cfg.n_steps = step
            cfg.l1_coeff = model.l1_coeff

            if resample_at_step_idx():
                current_frequency_counter = running_frequency_counter.to(cfg.device) / (cfg.batch_size * (step - last_resampling_step_idx))

                running_frequency_counter = torch.zeros_like(running_frequency_counter) # reset running counter to all zeros
                #from https://github.com/ArthurConmy/sae/tree/8bf510d9285eb5d79f77fe6896f2166d35f06a2b)
                fig = hist(
                    torch.max(current_frequency_counter.cpu(), torch.FloatTensor([1e-10])).log10().cpu(),
                    # Show proportion on y axis
                    histnorm="percent",
                    title = "Histogram of SAE Neuron Firing Frequency (Proportions of all Neurons)",
                    xaxis_title = "Log10(Frequency)",
                    yaxis_title = "Percent (changed!) of Neurons",
                    return_fig = True,
                    # Do not show legend
                    showlegend = False
                )
                metrics["frequency_histogram"] = fig

                last_resampling_step_idx = step

            wandb.log(metrics)

            if step % test_steps == 0:
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Testing"): # we only use fraction
                        batch = batch.to(model.cfg.device)
                        result = model.forward(batch, method="with_loss")
                        wandb.log(
                            {
                                f'test_{key}': value for key, value in result.format_data().items()
                            }
                        )

            cfg.n_epochs += 1


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
            all_tokens = load_tensor(file)
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



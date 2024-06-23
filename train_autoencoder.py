from common import (
    app, 
    vol, 
    image, 
    EMB_FOLDER,
    PATH, 
    LAION_DATASET_PATH,
    dataset_vol
)
import inspect
import time
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from bitsandbytes.optim import Adam8bit
from Laion_Processing.dataloader import load_loaders
import wandb
from autoencoder import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderBaseConfig,
    TopKAutoEncoderConfig,
    TopKAutoEncoder,
)
from modal import gpu
import modal
import os
from typing import Literal, Optional, Union, cast

from torch.optim import AdamW
from tqdm import tqdm
from training_config import AutoencoderTrainConfig
from utils import get_device
from _types import Loss_Method
from mechninterp_utils import hist, anthropic_resample #TODO MOVE THIS TO MECHINTERP UTILS
from torch.optim.lr_scheduler import StepLR  # Add this import



@app.function(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=10*60*60, #3 hours
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True
)
def train_autoencoder(
    train_cfg : AutoencoderTrainConfig,
    model_cfg : Union[AutoEncoderBaseConfig, TopKAutoEncoderConfig],
    retrain_path : Optional[str] = None
):
    if model_cfg is None:
        if retrain_path is None:
            raise ValueError("model_cfg or retrain_path must be provided")
        
        model_type = train_cfg.type
        if model_type == "autoencoder":
            model, train_cfg = AutoEncoder.load_from_checkpoint(retrain_path)
        elif model_type == "gated_autoencoder":
            model, train_cfg = GatedAutoEncoder.load_from_checkpoint(retrain_path)
        elif model_type == "topk_autoencoder":
            model, train_cfg = TopKAutoEncoder.load_from_checkpoint(retrain_path)
        else:
            raise ValueError(f"model_type {model_type} not recognized")
        model_dir = retrain_path
        print("retraining model")
        #TODO configure wandb to log to same project as original run

    else: 
        assert train_cfg.type == model_cfg.type, f"train_cfg.type {train_cfg.type} does not match model_cfg.type {model_cfg.type}"
        paths = [os.path.join(EMB_FOLDER, p) for p in os.listdir(EMB_FOLDER)]
        train_files = paths[:int(len(paths)*0.8)]
        test_files = paths[int(len(paths)*0.8):]
        
        if model_cfg.type == "autoencoder":
            model = AutoEncoder(model_cfg)
        elif model_cfg.type == "gated_autoencoder":
            model = GatedAutoEncoder(model_cfg)
        elif model_cfg.type == "topk_autoencoder":
            model = TopKAutoEncoder(model_cfg) #type: ignore
        else:
            raise ValueError(f"type {type} not recognized")
    
    def resample_at_step_idx()->bool:
        if train_cfg.resampling_interval is None:
            return False
        else:
            return step % train_cfg.resampling_interval == 0    

    def anthropic_resampling_at_step_idx()->bool:
        if train_cfg.anthropic_resampling and train_cfg.anthropic_resample_look_back_steps is not None:
            return step % train_cfg.anthropic_resample_look_back_steps == 0
        else:
            return False

    model_dir = f"{PATH}/laion2b_autoencoders/{model.cfg.folder_name}"
    os.makedirs(model_dir, exist_ok=True)
    vol.commit()

    if train_cfg.wandb_log:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(
            # set the wandb project where this run will be logged
            project="Sparse AutoEncoder",
            # track hyperparameters and run metadata
            name = model.cfg.folder_name,
            config={
                "learning_rate": train_cfg.lr,
                "architecture": train_cfg.type,
                "dataset": "Laion2B",
                "epochs": train_cfg.n_epochs,
            }
        )

    train_loader, test_loader = load_loaders(
        batch_size=(train_cfg.batch_size, 4*train_cfg.batch_size), 
        emb_folder=EMB_FOLDER,
        train_share=0.8,
        d_hidden = model.W_dec.shape[0], #TODO what is this variable again?
        n_counts=(None, 5)
    )

    model.to(train_cfg.device)
    fused_available = 'fused' in inspect.signature(AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    if train_cfg.adam8bit:
        optimizer = Adam8bit(model.parameters(), lr=train_cfg.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=train_cfg.lr, fused=use_fused)
    
    scheduler = StepLR(
        optimizer, step_size=train_cfg.batch_size*100, gamma=train_cfg.sched_lr_factor
    ) if train_cfg.sched_lr_factor is not None else None

    if retrain_path is None:
        step = 0
    else:
        step = model.metadata_cfg.n_steps + 1
        #this is to avoid resaving and reevaluating the same model
    
    last_resampling_step_idx = step

    #TODO think about effects of retraining here if model.cfg_metadata.l1_coeff is below max_l_coef
    l1_ramp = train_cfg.max_l_coef / ((train_cfg.n_steps // 100)*5 )
    # we linearly increase l1_coeff from 0 to 5 
    #over the first 5% of the total steps as per anthropic paper
    running_frequency_counter = torch.zeros(model.W_dec.shape[0], dtype=torch.int, device=model.cfg.device)
    ema_running_counter = torch.zeros(model.W_dec.shape[0], dtype=torch.float32, device=model.cfg.device)

    # Define the decay factor for the EMA
    decay = 0.99 

    #did_fire_last_10_k_steps = torch.zeros(model.W_dec.shape[0], dtype=torch.bool) #TODO is this needed?

    times = []
    with tqdm(total=train_cfg.n_steps, desc="Training Progress") as pbar:
        while step < train_cfg.n_steps:
            model.train()
            for batch in train_loader: 
                step += 1
                t0 = time.time()
                # Update l1_coeff linearly over the first 5% of the total steps
                if train_cfg.with_ramp:
                    if model.l1_coeff <= train_cfg.max_l_coef:
                        model.l1_coeff += l1_ramp 

                batch = batch.to(model.cfg.device)
                #we have scaled the embeddings by 10 to make them larger and easier to work with

                optimizer.zero_grad()
                #with torch.autocast(model.cfg.device):
                if train_cfg.type == "topk_autoencoder":
                    result = model.forward(
                        batch, 
                        method=train_cfg.loss_func, # type: ignore
                        ema_frequency_counter=ema_running_counter # type: ignore
                    ) 
                else:
                    result = model.forward(batch, method=train_cfg.loss_func) # type: ignore
                running_frequency_counter += result.did_fire

                # Update the EMA
                ema_running_counter = decay * ema_running_counter + (1 - decay) * running_frequency_counter.float()
                result.loss.backward()

                clip_grad_norm_(model.parameters(), max_norm=1) 
                model.remove_parallel_component_of_grads()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                if step % 1000 == 0:
                    print('step', step)
                    raise Exception('stop')

                torch.cuda.synchronize()
                times.append(time.time() - t0)
                if len(times) > 100:
                    mean_time = np.mean(times)
                    times = []
                else:
                    mean_time = np.mean(times) if times else None

                metrics = {
                    "l1_coeff": model.l1_coeff,
                    'lr': optimizer.param_groups[0]['lr'],
                    'time_per_step': mean_time,
                    **result.format_data()
                }

                with torch.no_grad():
                    # Renormalize W_dec to have unit norm
                    norms = torch.norm(model.W_dec.data, dim=1, keepdim=True)
                    model.W_dec.data /= (norms + 1e-6) 

                if step % train_cfg.save_interval == 0:
                    print("saving model at step", step)
                    model.save_model(train_cfg, model_dir)
                
                train_cfg.n_steps = step
                train_cfg.l1_coeff = model.l1_coeff

                if resample_at_step_idx():
                    #TODO use exponetial moving average(ema) instead??
                    current_frequency_counter = running_frequency_counter / (
                        train_cfg.batch_size * (
                        step - last_resampling_step_idx)
                    )

                    #from https://github.com/ArthurConmy/sae/tree/8bf510d9285eb5d79f77fe6896f2166d35f06a2b)
                    fig = hist(
                        torch.max(current_frequency_counter.cpu(), torch.FloatTensor([1e-10])).log10().cpu(),
                        # Show proportion on y axis
                        histnorm="percent",
                        title = "Histogram of SAE Neuron Firing Frequency (Proportions of all Neurons)",
                        xaxis_title = "Log10(Frequency)",
                        yaxis_title = "Percent (changed!) of Neurons",
                        return_fig = True,
                        showlegend = False
                    )
                    metrics["frequency_histogram"] = fig
                    if not train_cfg.anthropic_resampling:
                        running_frequency_counter = torch.zeros_like(running_frequency_counter) # reset running counter to all zeros
                
                if anthropic_resampling_at_step_idx():
                    print("resampling at step", step)
                    if train_cfg.wandb_log:
                        wandb.log({'resample_step' : step})
                    indices = (running_frequency_counter == 0).nonzero(as_tuple=False)[:, 0]
                    if len(indices) > 0:
                        resampling_stats = anthropic_resample(
                            indices=indices,
                            val_dataset=test_loader.dataloader,
                            model=model,
                            optimizer=optimizer,
                            sched = scheduler,
                            resampling_dataset_size=train_cfg.resampling_dataset_size,
                            resample_factor=0.2,
                            bias_resample_factor=0.2
                        )
                    
                        last_resampling_step_idx = step
                        running_frequency_counter = torch.zeros_like(running_frequency_counter)
                        torch.cuda.empty_cache()
                        metrics.update(resampling_stats.model_dump())
                    else:
                        print("no indices to resample")

                if train_cfg.wandb_log:
                    wandb.log(metrics)
                
                pbar.update(1)

                if step % train_cfg.test_steps == 0:
                    model.eval()
                    with torch.no_grad():
                        for batch in tqdm(test_loader, desc="Testing"): # we only use fraction
                            batch = batch.to(model.cfg.device)
                            if isinstance(model, TopKAutoEncoder):
                                result = model.forward(batch, method="with_loss", ema_frequency_counter=ema_running_counter)
                            else:
                                result = model.forward(batch, method="with_loss")
                            if train_cfg.wandb_log:
                                wandb.log(
                                    {
                                    f'test_{key}': value for key, value in result.format_data().items()
                                }) 
                            else:
                                print(result.format_data())
                train_cfg.n_epochs += 1


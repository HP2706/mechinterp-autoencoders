from common import (
    stub, 
    vol, 
    image, 
    EMB_FOLDER,
    PATH, 
    LAION_DATASET_PATH,
    dataset_vol
)
import time
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from Laion_Processing.dataloader import load_loaders
import wandb
from autoencoder import (
    AutoEncoderBase, 
    AutoEncoder,
    AutoencoderResult,
    AutoencoderTrainConfig, 
    GatedAutoEncoder,
    GatedAutoEncoderResult, 
    TopKAutoEncoder,
    TopKAutoEncoderModelConfig
)
from modal import gpu
import modal
import os
from typing import Literal, Optional, Union

from torch.optim import AdamW
from tqdm import tqdm
from utils import get_device
from _types import Loss_Method
from mechninterp_utils import hist, anthropic_resample #TODO MOVE THIS TO MECHINTERP UTILS
from torch.optim.lr_scheduler import StepLR  # Add this import

def save_model(
    model : Union[GatedAutoEncoder, AutoEncoder], 
    cfg: AutoencoderTrainConfig,
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
    timeout=10*60*60, #3 hours
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True
)
def train_autoencoder(
    type: Literal['autoencoder', 'gated_autoencoder', 'topk_autoencoder'], 
    dict_mult : int,
    anthropic_resampling : bool = False,
    anthropic_resample_look_back_steps : int = 12500,#anthropic paper uses this number
    resampling_dataset_size : int = 819200, #anthropic paper uses this number
    resampling_interval : Optional[int] = int(2.5*10**4),
    steps: int = 10**5, # 100 k steps as per anthropic paper
    save_interval : int = 10**4, # we save the model every save_interval steps
    test_steps : int = 25*10**3,
    with_ramp: bool = True,
    max_l_coef : float = 5,
    loss_func: Loss_Method = 'with_loss',
    retrain_path : Optional[str] = None,
    sched_lr_factor : Optional[float] = None,
    wandb_log : bool = True
):
    if with_ramp and loss_func == 'with_loss':
        print("with_ramp does not work for the with_loss loss function disabling it")
        with_ramp = False

    if retrain_path is not None:
        model = AutoEncoderBase.load_from_checkpoint(retrain_path)
        model_dir = retrain_path
        print("retraining model")
        print("additional training steps", steps-model.metadata_cfg.n_steps)
        cfg = model.metadata_cfg
    else:
        d_mlp = 768 
        paths = [os.path.join(EMB_FOLDER, p) for p in os.listdir(EMB_FOLDER)]
        train_files = paths[:int(len(paths)*0.8)]
        test_files = paths[int(len(paths)*0.8):]

        cfg = AutoencoderTrainConfig(
            wandb_log=wandb_log,
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
            updated_anthropic_method=True if loss_func=='with_new_loss' else False,
            anthropic_resampling=anthropic_resampling,
            anthropic_resample_look_back_steps=anthropic_resample_look_back_steps if anthropic_resampling else None,
            sched_lr_factor=sched_lr_factor
        )

        if type == "gated_autoencoder":
            model = GatedAutoEncoder(cfg)
        elif type == "autoencoder":
            model = AutoEncoder(cfg)
        elif type == "topk_autoencoder":
            model_cfg = TopKAutoEncoderModelConfig(
                k = 10, #TODO let user set
                **cfg.model_dump()
            )
            model = TopKAutoEncoder(model_cfg)
        else:
            raise ValueError(f"type {type} not recognized")
    
    model = torch.compile(model)

    def resample_at_step_idx()->bool:
        if resampling_interval is None:
            return False
        else:
            return step % resampling_interval == 0    

    def anthropic_resampling_at_step_idx()->bool:
        if anthropic_resampling:
            return step % anthropic_resample_look_back_steps == 0
        else:
            return False

    model_dir = f"{PATH}/laion2b_autoencoders/{cfg.folder_name}"
    os.makedirs(model_dir, exist_ok=True)
    vol.commit()

    if wandb_log:
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

    train_loader, test_loader = load_loaders(
        batch_size=(cfg.batch_size, 4*cfg.batch_size), 
        emb_folder=EMB_FOLDER,
        train_share=0.8,
        d_hidden = model.W_dec.shape[0] if cfg.updated_anthropic_method else None,
        n_counts=(None, 5)
    )

    model.to(model.cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    
    scheduler = StepLR(
        optimizer, step_size=cfg.batch_size*100, gamma=sched_lr_factor
    ) if sched_lr_factor is not None else None

    if retrain_path is None:
        step = 0
    else:
        step = model.metadata_cfg.n_steps + 1
        #this is to avoid resaving and reevaluating the same model
    
    last_resampling_step_idx = step

    #TODO think about effects of retraining here if model.cfg_metadata.l1_coeff is below max_l_coef
    l1_ramp = max_l_coef / ((steps // 100)*5 )
    # we linearly increase l1_coeff from 0 to 5 
    #over the first 5% of the total steps as per anthropic paper

    running_frequency_counter = torch.zeros(model.W_dec.shape[0], dtype=torch.int)

    times = []
    while step < steps:
        model.train()
        for batch in tqdm(train_loader, total = len(train_loader), desc="dataset training"): 
            step += 1
            t0 = time.time()
            # Update l1_coeff linearly over the first 5% of the total steps
            if with_ramp:
                if model.l1_coeff <= max_l_coef:
                    model.l1_coeff += l1_ramp 

            batch = batch.to(model.cfg.device)
            #we have scaled the embeddings by 10 to make them larger and easier to work with

            optimizer.zero_grad()
            with torch.autocast(model.cfg.device):
                result : Union[AutoencoderResult, GatedAutoEncoderResult] = model.forward(batch, method=loss_func) # type: ignore
            running_frequency_counter += result.did_fire

            result.loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1) 
            # clip grad norm as per anthropic https://transformer-circuits.pub/2024/april-update/index.html#training-saes 
            
            model.remove_parallel_component_of_grads()
            optimizer.step()
            if scheduler:
                scheduler.step()

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
                current_frequency_counter = running_frequency_counter.to(cfg.device) / (
                    cfg.batch_size * (
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
                if not anthropic_resampling:
                    running_frequency_counter = torch.zeros_like(running_frequency_counter) # reset running counter to all zeros
            
            if anthropic_resampling_at_step_idx():
                print("resampling at step", step)
                if cfg.wandb_log:
                    wandb.log({'resample_step' : step})
                indices = (running_frequency_counter == 0).nonzero(as_tuple=False)[:, 0]
                if len(indices) > 0:
                    resampling_stats = anthropic_resample(
                        indices=indices,
                        val_dataset=test_loader.dataloader,
                        model=model,
                        optimizer=optimizer,
                        sched = scheduler,
                        resampling_dataset_size=resampling_dataset_size,
                        resample_factor=0.2,
                        bias_resample_factor=0.2
                    )
                
                    last_resampling_step_idx = step
                    running_frequency_counter = torch.zeros_like(running_frequency_counter)
                    torch.cuda.empty_cache()
                    metrics.update(resampling_stats.model_dump())
                else:
                    print("no indices to resample")

            if cfg.wandb_log:
                wandb.log(metrics)

            if step % test_steps == 0:
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Testing"): # we only use fraction
                        batch = batch.to(model.cfg.device)
                        result : Union[AutoencoderResult, GatedAutoEncoderResult] = model.forward(batch, method="with_loss")
                        if cfg.wandb_log:
                            wandb.log(
                                {
                                f'test_{key}': value for key, value in result.format_data().items()
                            }) 
                        else:
                            print(result.format_data())
            cfg.n_epochs += 1

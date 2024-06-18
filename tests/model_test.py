from typing import List, Literal, Tuple, Type, Union
import shutil
import unittest
from tqdm import tqdm
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from training_config import AutoencoderTrainConfig
from autoencoder import (
    AutoEncoder,
    AutoEncoderBaseConfig,
    GatedAutoEncoder,
    TopKAutoEncoder,
    TopKAutoEncoderConfig
)

def generate_random_frequency_counter(size, decay=0.9):
    ema = torch.zeros(size)
    # Simulate sparse activations
    for _ in range(100):  # Number of iterations to simulate
        activations = torch.randint(0, 2, (size,))  # Random sparse activations (0 or 1)
        ema = decay * ema + (1 - decay) * activations
    return ema

def run_nan_check_test(autoencoder_class, config_instance):
    n_steps = 100
    d_input = config_instance.d_input
    n_tries = 10
    print(autoencoder_class.__name__, autoencoder_class, config_instance)

    if isinstance(config_instance, TopKAutoEncoderConfig):
        ema_frequency_counter = generate_random_frequency_counter(d_input).to(config_instance.device)

    for _try in tqdm(range(n_tries)):
        model = autoencoder_class(config_instance)
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN after initialization found in param for {name}, {param}"

        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data = torch.randn((10, d_input)).to(model.cfg.device).to(model.W_dec.dtype)
        if torch.isnan(data).any():
            raise ValueError("NaN found in data")

        for _ in range(n_steps):
            if isinstance(config_instance, TopKAutoEncoderConfig):
                res = model.forward(
                    data, 
                    method='with_loss', 
                    ema_frequency_counter=ema_frequency_counter #type: ignore
                )
            else:
                res = model.forward(data, method='with_loss')
            optim.zero_grad()

            if torch.isnan(res.loss).any():
                raise ValueError("NaN found in loss")
            
            res.loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Gradient for {name} is None, skipping NaN check for this parameter.")
                    continue
                assert not torch.isnan(param.grad).any(), f"NaN found in grad for {name}, {param.grad}"

            optim.step()

            for name, param in model.named_parameters():
                if param is None:
                    print(f"param {name} is None is this Intentional??")
                    continue
                assert not torch.isnan(param).any(), f"NaN found in param for {name}, {param}"

@pytest.mark.usefixtures
def test_check_nans():
    devices : List[Literal['cpu', 'cuda', 'mps']] = ['cpu'] 
    if torch.cuda.is_available():
        devices.append('cuda')
    elif torch.backends.mps.is_available():
        devices.append('mps')

    autoencoder_configs : List[
        Tuple[
            Union[Type[TopKAutoEncoder], Type[GatedAutoEncoder], Type[AutoEncoder]], 
            AutoEncoderBaseConfig
        ]
    ] = []

    for device in devices:
        autoencoder_configs.append(
            (TopKAutoEncoder, TopKAutoEncoderConfig(
            seed=42,
            l1_coeff=10e-3, 
            dict_mult=2,
            d_input=1,
            device=device, 
            type='gated_autoencoder',
            k=1,
            k_aux=1,
            updated_anthropic_method=True,
            ))
        )
        autoencoder_configs.append(
            (GatedAutoEncoder, AutoEncoderBaseConfig(
                seed=42,
                l1_coeff=10e-3, 
                dict_mult=2,
                d_input=1,
                device=device, 
                type='gated_autoencoder',
                updated_anthropic_method=True,
            ))
        )
        autoencoder_configs.append(
            (AutoEncoder, AutoEncoderBaseConfig(
            seed=42,
            l1_coeff=10e-3, 
            dict_mult=2,
            d_input=1,
            device=device, 
            type='autoencoder',
            updated_anthropic_method=True,
            ))
        )
 
    for autoencoder_class, config_instance in autoencoder_configs:
        print(f"Running nan check test for {autoencoder_class.__name__} on {config_instance.device}")
        run_nan_check_test(autoencoder_class, config_instance)


@pytest.mark.usefixtures
def test_load_from_checkpoint():
    model = TopKAutoEncoder(TopKAutoEncoderConfig(
        seed=42,
        l1_coeff=10e-3, 
        dict_mult=2,
        d_input=1,
        device='cpu', 
        type='gated_autoencoder',
        k=1,
        k_aux=1,
        updated_anthropic_method=True,
    ))
    model.save_model(AutoencoderTrainConfig.dummy_default(), 'test_model_dir')
    model.load_from_checkpoint(model.model_path)
    shutil.rmtree('test_model_dir')

if __name__ == '__main__':
    pytest.main()
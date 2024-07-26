import sys
import os
import random
from pytest import Function
import itertools
from regex import F
import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time
import pandas as pd
import plotly.express as px
from mechinterp_autoencoders.base_autoencoder import BaseAutoEncoder
from mechinterp_autoencoders.GatedAutoencoder import GatedAutoEncoderConfig, GatedAutoEncoder
from mechinterp_autoencoders.autoencoder import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders.jump_relu import JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig
from mechinterp_autoencoders.topk_autoencoder import TopKAutoEncoder, TopKAutoEncoderConfig
from mechinterp_autoencoders.utils import generate_sparse_tensor

def test_autoencoder_benchmark(
    model_cls, 
    config_cls, 
    dict_mult, 
    d_input, 
    k, 
    sparsity_level, 
    use_kernel, 
    batch_size
):
    if model_cls == TopKAutoEncoder:
        cfg = TopKAutoEncoderConfig(
            dict_mult=dict_mult,
            d_input=d_input,
            k=k,
            k_aux=k,
            use_kernel=use_kernel
        )
    elif model_cls == JumpReLUAutoEncoder:
        cfg = JumpReLUAutoEncoderConfig(
            dict_mult=dict_mult,
            d_input=d_input,
            use_kernel=use_kernel,
            l1_coeff=0.01,
            threshold=sparsity_level
        )
    elif model_cls in [AutoEncoder, GatedAutoEncoder]:
        cfg = config_cls(
            dict_mult=dict_mult,
            d_input=d_input,
            use_kernel=use_kernel,
            l1_coeff=0.01,
        )
    else:
        raise ValueError(f'Unknown model class: {model_cls}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_cls(cfg).to(device)
    
    total_time = 0
    num_iterations = 10
    
    with torch.autocast(
        device_type=device, 
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32
    ):
        gpu_memory_usage = []
        for _ in range(num_iterations):
            x = generate_sparse_tensor(batch_size, d_input, sparsity_level).to(device)
            start_time = time.time()
            if isinstance(model, TopKAutoEncoder):
                model.forward(x, method='with_loss', ema_frequency_counter=torch.randn(d_input).to(device))
            else:
                model.forward(x, method='with_loss')
            gpu_memory_usage.append(torch.cuda.memory_allocated(device))
            end_time = time.time()
            total_time += end_time - start_time
    
    avg_time = total_time / num_iterations
    avg_gpu_memory_usage = sum(gpu_memory_usage) / num_iterations

    assert True
    return {
        'model': model_cls.__name__,
        'dict_mult': dict_mult,
        'd_input': d_input,
        'k': k,
        'sparsity_level': sparsity_level,
        'use_kernel': use_kernel,
        'batch_size': batch_size,
        'avg_time': avg_time,
        'avg_gpu_memory_usage': avg_gpu_memory_usage
    }

def visualize_and_save(df: pd.DataFrame, save_path: str):
    
    #NOTE DONT CHANGE THIS PATH
    os.makedirs(save_path, exist_ok=True)
    df.to_parquet(f"{save_path}/benchmark_results.parquet")
    print(f"saving to {save_path}/benchmark_results.parquet")
    # Group by all parameters except d_input and avg_time
    grouping_cols = ['model', 'dict_mult', 'k', 'sparsity_level', 'batch_size', 'use_kernel']
    
    for use_kernel in [True, False]:
        fig = px.line(df[df['use_kernel'] == use_kernel], x='d_input', y='avg_time',
                color='model', line_dash='dict_mult', symbol='k',
                facet_col='sparsity_level', facet_row='batch_size',
                hover_data=grouping_cols,
                title=f'use_kernel: {use_kernel} Performance Curves by Model and Input Dimension')
        fig.update_layout(height=1000, width=1200)
        fig.write_html(f'{save_path}/performance_curves{"with_kernel" if use_kernel else ""}.html')
        fig.show()

def get_params(
    models: list[type[BaseAutoEncoder]],
    local_test: bool = False
) -> list[list[dict]]:

    cfg_dict = {
        AutoEncoder: AutoEncoderConfig,
        GatedAutoEncoder: GatedAutoEncoderConfig,
        TopKAutoEncoder: TopKAutoEncoderConfig,
        JumpReLUAutoEncoder: JumpReLUAutoEncoderConfig
    }

    model_cfgs = [
        (model, cfg_dict[model])
        for model in models
    ]
    
    if local_test:
        param_values = [
            model_cfgs,
            [1],  # dict_mults
            [768],  # d_inputs
            [8],  # ks
            [0.001],  # sparsity_levels
            [False],  # use_kernels
            [10]  # batch_sizes
        ]
    else:
        param_values = [
            model_cfgs,
            [16],  # dict_mults
            [768],  # d_inputs
            [8],  # ks
            [0.1, 0.001],  # sparsity_levels
            [True, False],  # use_kernels
            [128]  # batch_sizes
        ]

    param_names = ['model_config', 'dict_mult', 'd_input', 'k', 'sparsity_level', 'use_kernel', 'batch_size']

    all_params = []
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        param_dict['model_cls'], param_dict['config_cls'] = param_dict.pop('model_config')
        all_params.append(param_dict)

    random.shuffle(all_params)
    print('all_params', all_params)
    all_params = [list(all_params)[i:i+10] for i in range(0, len(all_params), 10)]
    return all_params

def run_benchmarks(
    save_path: str,
    models: list[type[BaseAutoEncoder]],
    local_test: bool = False
):
    results = []
    all_params = get_params(models, local_test)
    for param_list in tqdm.tqdm(all_params):
        #we use a list to get better sense of progress
        for params in param_list:
            result = test_autoencoder_benchmark(**params)
            results.append(result)

    visualize_and_save(pd.DataFrame(results), save_path)

if __name__ == '__main__':
    os.makedirs('benchmarks/data', exist_ok=True)
    run_benchmarks('benchmarks/data', [AutoEncoder, TopKAutoEncoder], local_test=True)
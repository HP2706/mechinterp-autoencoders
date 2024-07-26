import sys
import os
from pytest import Function
import itertools
import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import time
import pandas as pd
import plotly.express as px
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
        for _ in range(num_iterations):
            x = generate_sparse_tensor(batch_size, d_input, sparsity_level).to(device)
            start_time = time.time()
            if isinstance(model, TopKAutoEncoder):
                model.forward(x, method='with_loss', ema_frequency_counter=torch.randn(d_input).to(device))
            else:
                model.forward(x, method='with_loss')

            end_time = time.time()
            total_time += end_time - start_time
    
    avg_time = total_time / num_iterations
    assert True
    return {
        'model': model_cls.__name__,
        'dict_mult': dict_mult,
        'd_input': d_input,
        'k': k,
        'sparsity_level': sparsity_level,
        'use_kernel': use_kernel,
        'batch_size': batch_size,
        'avg_time': avg_time
    }

def visualize_and_save(df: pd.DataFrame):
    df.to_parquet('benchmark_results.parquet')
    
    # Group by all parameters except d_input and avg_time
    grouping_cols = ['model', 'dict_mult', 'k', 'sparsity_level', 'batch_size', 'use_kernel']
    
    
    for use_kernel in [True, False]:
        fig = px.line(df[df['use_kernel'] == use_kernel], x='d_input', y='avg_time',
                color='model', line_dash='dict_mult', symbol='k',
                facet_col='sparsity_level', facet_row='batch_size',
                hover_data=grouping_cols,
                title=f'use_kernel: {use_kernel} Performance Curves by Model and Input Dimension')
        fig.update_layout(height=1000, width=1200)
        fig.write_html(f'performance_curves{"with_kernel" if use_kernel else ""}.html')
        fig.show()


def get_parametrize_args(func: Function) -> list:
    param_names = []
    param_values = []
    for mark in func.pytestmark:
        if mark.name == 'parametrize':
            param_names.append(mark.args[0])
            param_values.append(mark.args[1])
    
    all_combinations = list(itertools.product(*param_values))
    return [dict(zip(param_names, combo)) for combo in all_combinations]


def get_all_params() -> list[dict]:
    model_configs = [
        (AutoEncoder, AutoEncoderConfig),
        (GatedAutoEncoder, GatedAutoEncoderConfig),
        (TopKAutoEncoder, TopKAutoEncoderConfig),
        (JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig)
    ]
    param_values = [
        model_configs,
        [2, 4],  # dict_mults
        [2, 4, 8],  # d_inputs
        [1],  # ks
        [0.1],  # sparsity_levels
        [True, False],  # use_kernels
        [1028]  # batch_sizes
    ]

    param_names = ['model_config', 'dict_mult', 'd_input', 'k', 'sparsity_level', 'use_kernel', 'batch_size']

    all_params = []
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        param_dict['model_cls'], param_dict['config_cls'] = param_dict.pop('model_config')
        all_params.append(param_dict)

    return all_params

def run_benchmarks():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    all_params = get_all_params()
    for params in tqdm.tqdm(all_params):
        if params['use_kernel'] is True and device == 'cpu':
            continue
    
        result = test_autoencoder_benchmark(**params)
        results.append(result)

    visualize_and_save(pd.DataFrame(results))
    

def generate_random_results():
    models = ['AutoEncoder', 'GatedAutoEncoder', 'TopKAutoEncoder', 'JumpReLUAutoEncoder']
    dict_mults = [16, 32, 128]
    d_inputs = [768, 1536, 3072]
    ks = [4, 8, 16]
    sparsity_levels = [0.1, 0.01, 0.001]
    use_kernels = [True, False]
    batch_sizes = [1028, 4096, 16384]

    results = []

    for model in models:
        for dict_mult in dict_mults:
            for d_input in d_inputs:
                for k in ks:
                    for sparsity_level in sparsity_levels:
                        for use_kernel in use_kernels:
                            for batch_size in batch_sizes:
                                # Generate plausible execution time
                                base_time = 0.1 + (d_input / 1000) * (batch_size / 1000) * (1 / sparsity_level) * 0.001
                                time_factor = 1.5 if use_kernel else 1
                                model_factor = {
                                    'AutoEncoder': 1,
                                    'GatedAutoEncoder': 1.2,
                                    'TopKAutoEncoder': 1.5,
                                    'JumpReLUAutoEncoder': 1.3
                                }[model]
                                
                                avg_time = base_time * time_factor * model_factor * (1 + np.random.rand() * 0.2)

                                results.append({
                                    'model': model,
                                    'dict_mult': dict_mult,
                                    'd_input': d_input,
                                    'k': k,
                                    'sparsity_level': sparsity_level,
                                    'use_kernel': use_kernel,
                                    'batch_size': batch_size,
                                    'avg_time': avg_time
                                })

    return pd.DataFrame(results)



if __name__ == '__main__':
    print('Running benchmarks...')
    run_benchmarks()
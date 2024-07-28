import sys
import os
import random
from typing import Callable
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
from mechinterp_autoencoders.utils import generate_sparse_tensor, get_device, extract_nonzero

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
    if model_cls.__name__ == 'TopKAutoEncoder':
        cfg = TopKAutoEncoderConfig(
            dict_mult=dict_mult,
            d_input=d_input,
            k=k,
            k_aux=k,
            use_kernel=use_kernel
        )
    elif model_cls.__name__ == 'JumpReLUAutoEncoder':
        cfg = JumpReLUAutoEncoderConfig(
            dict_mult=dict_mult,
            d_input=d_input,
            l1_coeff=0.01,
            threshold=sparsity_level
        )
    elif model_cls.__name__ in ['AutoEncoder', 'GatedAutoEncoder']:
        cfg = config_cls(
            dict_mult=dict_mult,
            d_input=d_input,
            l1_coeff=0.01,
        )
    else:
        raise ValueError(f'Unknown model class: {model_cls}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_cls(cfg).to(device)
    
    total_time = 0
    num_iterations = 3
    
    with torch.autocast(
        device_type=device.type, 
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32
    ):
        gpu_memory_usage = []
        t0 = time.time()
        x = generate_sparse_tensor((batch_size, d_input), sparsity_level, device)
        print(f'generate_sparse_tensor took {time.time() - t0} seconds')
        for _ in range(num_iterations):
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
    
    # Apply negative reciprocal transformation to avg_time
    df['avg_time'] = df['avg_time'] * 1000
    df['log_avg_time'] = np.log10(df['avg_time'])
    for use_kernel in [True, False]:
        fig = px.line(df, x='dict_mult', y='log_avg_time',
            color=df.apply(lambda row: f"{row['model']} ({'with' if row['use_kernel'] else 'without'} kernel)", axis=1),
            line_dash='d_input', symbol='k',
            facet_col='sparsity_level', facet_row='batch_size',
            hover_data=grouping_cols + ['avg_time'],
            title=('Performance Curves by Model and Input Dimension (Transformed Scale)'),
            height=1000, 
            width=1200
        )
        
        # Update y-axis to show original values
        max_time = df['log_avg_time'].max()
        tick_values = [0.001, 0.01, 0.1, min(1, max_time)]
        fig.update_layout(yaxis=dict(
            tickmode='array',
            tickvals=[-1/t for t in tick_values],
            ticktext=[f'{t:.3f}' for t in tick_values],
            title='log(avg_time) (milliseconds)'
        ))

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
            [16, 32, 64],  # dict_mults
            [768],  # d_inputs
            [8, 16, 32],  # ks
            [0.1, 0.001, 0.0001, 0.00001],  # sparsity_levels
            [True, False],  # use_kernels
            [1024, 2048]  # batch_sizes
        ]

    param_names = ['model_config', 'dict_mult', 'd_input', 'k', 'sparsity_level', 'use_kernel', 'batch_size']

    all_params = []
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        param_dict['model_cls'], param_dict['config_cls'] = param_dict.pop('model_config')
        all_params.append(param_dict)

    random.shuffle(all_params)
    all_params = [list(all_params)[i:i+10] for i in range(0, len(all_params), 10)]
    return all_params

def benchmark_decode(save_path: str, decode_funcs: dict[str, callable]):
    results = []
    
    dict_mults = [32, 64, 128, 256]
    d_inputs = [768]
    sparsity_levels = [0.001, 0.0001]
    batch_sizes = [1024]
    
    device = get_device()
    params = itertools.product(dict_mults, d_inputs, sparsity_levels, batch_sizes)
    
    for dict_mult, d_input, sparsity_level, batch_size in tqdm.tqdm(list(params)):
        d_sae = d_input * dict_mult
        x = generate_sparse_tensor((batch_size, d_sae), sparsity_level, device)
        print("non-zero elms", torch.count_nonzero(x).item())

        autoencoder = AutoEncoder(
            AutoEncoderConfig(
                dict_mult=dict_mult, 
                d_input=d_input, 
                l1_coeff=0.01
            )
        ).to(device)

        for decode_name, decode_func in decode_funcs.items():

            decode_times = []
            for _ in range(2):  # Perform each measurement twice
                t0 = time.time()
                x_recons = decode_func(x, autoencoder)
                decode_time = time.time() - t0
                decode_times.append(decode_time)
            
            avg_decode_time = sum(decode_times) / len(decode_times)
            
            results.append({
                'dict_mult': dict_mult,
                'd_input': d_input,
                'sparsity_level': sparsity_level,
                'batch_size': batch_size,
                'decode_method': decode_name,
                'decode_time': avg_decode_time
            })
            
            print(f'\nAverage time for {decode_name} decode, {dict_mult} dict_mult, {d_input} d_input, {sparsity_level} sparsity, {batch_size} batch_size: {avg_decode_time:.4f} seconds\n\n')

    df = pd.DataFrame(results)
    os.makedirs(save_path, exist_ok=True)
    df.to_parquet(f"{save_path}/decode_benchmark_results.parquet")
    print(f"Saved results to {save_path}/decode_benchmark_results.parquet")

    # Visualize results
    fig = px.line(df, x='dict_mult', y='decode_time', color='decode_method',
                  facet_col='sparsity_level', facet_row='batch_size',
                  hover_data=['d_input', 'dict_mult'],
                  title='Decode Performance by Method and Parameters',
                  labels={'decode_time': 'Decode Time (seconds)'},
                  height=800, width=1200)
    
    fig.write_html(f'{save_path}/decode_performance_curves.html')
    fig.show()

def benchmark_get_nonzero(non_zero_fns: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]):
    dict_mults = [1024, 2048]
    d_inputs = [3072]
    sparsity_levels = [0.0001]
    batch_sizes = [1024, 2048, 4096, 8192]

    for fn_name, fn in non_zero_fns.items():
        for dict_mult, d_input, sparsity_level, batch_size in itertools.product(dict_mults, d_inputs, sparsity_levels, batch_sizes):
            d_sae = d_input * dict_mult
            x = generate_sparse_tensor((batch_size, d_sae), sparsity_level, get_device())
            t0 = time.perf_counter()
            fn(x)
            print(f'{fn_name} took {time.perf_counter() - t0} for {batch_size} batch_size, {d_input} d_input, {dict_mult} dict_mult, {k} k, {sparsity_level} sparsity')


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

def test():
    from mechinterp_autoencoders.utils import generate_sparse_tensor
    def eager_decode(x, autoencoder : AutoEncoder):
        top_vals, top_idx = extract_nonzero(x)
        return autoencoder.eager_decode(top_idx, top_vals)

    def kernel_decode(x, autoencoder : AutoEncoder):
        top_vals, top_idx = torch.topk(x, k=10, dim=1)
        return autoencoder.kernel_decode(top_idx, top_vals)

    def base_decode(x, autoencoder : AutoEncoder):
        return x @ autoencoder.W_dec

    decode_funcs = {
        #'eager': eager_decode, 
        'base': base_decode,
        'kernel': kernel_decode
    }

    #benchmark_decode('benchmarks/data', decode_funcs)
    run_benchmarks('benchmarks/data', [AutoEncoder, TopKAutoEncoder])

if __name__ == '__main__':
    #os.makedirs('benchmarks/data', exist_ok=True)
    test()
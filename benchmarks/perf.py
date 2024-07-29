import sys
import os
from typing import Any, List, Literal, Optional, cast
import itertools
import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from mechinterp_autoencoders.base_autoencoder import BaseAutoEncoder, AutoEncoderBaseConfig
from mechinterp_autoencoders.GatedAutoencoder import GatedAutoEncoderConfig, GatedAutoEncoder
from mechinterp_autoencoders.autoencoder import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders.jump_relu import JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig
from mechinterp_autoencoders.topk_autoencoder import TopKAutoEncoder, TopKAutoEncoderConfig
from mechinterp_autoencoders.utils import generate_sparse_tensor, get_device, extract_nonzero
from functools import wraps
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    from mechinterp_autoencoders.kernels import TritonDecoder

def shape_params(params: dict[str, list]) -> list[dict[str, Any]]:
    all_params = []
    for values in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), values))
        all_params.append(param_dict)
    return all_params

def benchmark(func=None, *, n_runs=2, with_memory=True):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            torch.cuda.empty_cache()
            
            total_time = 0
            results = []
            
            for i in range(n_runs+1):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                result = f(*args, **kwargs)
                end.record()
                
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                if i == 0:
                    #warmup first round doesnt count
                    total_time += elapsed_time
                    results.append(result)
                torch.cuda.empty_cache()

            avg_time = total_time / n_runs

            if with_memory:
                memory_allocated = torch.cuda.memory_allocated()
                return results[-1], avg_time, memory_allocated
            return results[-1], avg_time

        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

@benchmark(with_memory=False)
@torch.compile
def extract_nonzero_1(x):
    max_non_zero_elms = int((x != 0).any(dim=0).sum())
    _, top_indices = x.abs().topk(max_non_zero_elms, sorted=False)
    top_acts = x.gather(dim=-1, index=top_indices)

@benchmark(with_memory=False)
@torch.compile
def extract_nonzero_2(x):
    max_non_zero_elms = int((x != 0).sum(dim=-1).max())
    _, top_indices = x.abs().topk(max_non_zero_elms, sorted=False)
    top_acts = x.gather(dim=-1, index=top_indices)

@benchmark(with_memory=False)
@torch.compile
def extract_nonzero_3(x):
    _, top_indices = x.abs().topk(64, sorted=False)
    return x.gather(dim=-1, index=top_indices)

@benchmark(with_memory=False)
@torch.compile
def fixed_extract_nonzero(x):
    return torch.topk(x, 64, dim=1, sorted=False)

def benchmark_nonzero(dim_range, dict_mult=64, num_runs=10):
    results = {
        'dim': [],
        'extract_nonzero_1': [],
        'extract_nonzero_2': [],
        'extract_nonzero_3': [],
        'fixed_extract_nonzero': []
    }

    for d in dim_range:
        times = {k: [] for k in results.keys() if k != 'dim'}

        for _ in range(num_runs):
            x = generate_sparse_tensor((8, d*dict_mult), 0.00001, device=get_device())
            W_dec = torch.nn.Parameter(torch.randn(dict_mult*d, 128, device=get_device()).contiguous())

            _, times['extract_nonzero_1'].append(extract_nonzero_1(x)[1])
            _, times['extract_nonzero_2'].append(extract_nonzero_2(x)[1])
            _, times['extract_nonzero_3'].append(extract_nonzero_3(x)[1])
            _, times['fixed_extract_nonzero'].append(fixed_extract_nonzero(x)[1])
            torch.cuda.empty_cache()

        results['dim'].append(d)
        for k, v in times.items():
            results[k].append(np.mean(v))

    return pd.DataFrame(results)


def benchmark_models(
    save_path: str,
    models: list[tuple[type[BaseAutoEncoder], AutoEncoderBaseConfig]],
    interval_dicts: dict[str, list],
    num_runs: int = 3
):
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    @benchmark(with_memory=True, n_runs=num_runs)
    def benchmark_model(
        model: BaseAutoEncoder, 
        x: torch.Tensor, 
        method : Literal['reconstruct', 'with_loss']
    ):
        return model.forward(x, method=method)

    all_results = []

    for model_cls, config in tqdm.tqdm(models, desc="Models"):
        model_name = model_cls.__name__
        data = {}
        for params in tqdm.tqdm(shape_params(interval_dicts), desc="Params", leave=False):
            
            #set config NOTE this is bad
            config.dict_mult = params['dict_mult']
            config.d_input = params['dim']
            config.use_kernel = params['use_kernel']

            model = model_cls(config).to(get_device())
            model = torch.compile(model) if params['use_torch_compile'] else model
            model = cast(BaseAutoEncoder, model)

            x = generate_sparse_tensor(
                (params['batch_size'], params['dim']), 
                params['sparsity_level'], 
                device=get_device()
            )
            
            _, time, memory = benchmark_model(model, x, method=params['method'])

            # Combine all parameters, results, and model info into a single dictionary
            result = {
                'model': model_name,
                'time': time,
                **params  
            }
            all_results.append(result)

    df = pd.DataFrame(all_results)
    
    os.makedirs(save_path, exist_ok=True)
    df.to_parquet(f"{save_path}/benchmark_results.parquet")
    return df

def benchmark_decode(dim_range : List[int] , dict_mult : int = 64, num_runs : int = 5):
    #benchmark decode with backward pass
    @benchmark(with_memory=False, n_runs=num_runs)
    def kernel_decode(x : torch.Tensor, d : int, k : Optional[int] = None, y : Optional[torch.Tensor] = None):
        W_dec = nn.Parameter(torch.randn(dict_mult*d, d, device=get_device()).contiguous())
        top_vals, top_idx = extract_nonzero(x, k)
        out = TritonDecoder.apply(top_idx, top_vals, W_dec.mT)
        if y is not None:
            mse = (out - y).pow(2).mean()
            mse.backward()
        return out

    @benchmark(with_memory=False, n_runs=5)
    def base_decode(x : torch.Tensor, d : int, y : Optional[torch.Tensor] = None):
        W_dec = nn.Parameter(torch.randn(dict_mult*d, d, device=get_device()).contiguous())
        out = x @ W_dec
        if y is not None:
            mse = (out - y).pow(2).mean()
            mse.backward()
        return out
    
    results = {
        'dim': [],
        'base_decode': [],
        'kernel_unknown_k_decode': [],
        'kernel_known_k_decode': [],
    }
    for d in tqdm.tqdm(dim_range, desc="Dimensions"):
        print(f"Benchmarking dimension {d}")
        x = generate_sparse_tensor((8, d*dict_mult), 0.00001, device=get_device())
        _, base_time = base_decode(x, d)
        _, unknown_k_time = kernel_decode(x, d)
        _, known_k_time = kernel_decode(x, d, k=32)

        results['dim'].append(d*dict_mult)
        results['base_decode'].append(base_time)
        results['kernel_unknown_k_decode'].append(unknown_k_time)
        results['kernel_known_k_decode'].append(known_k_time)
    
    return pd.DataFrame(results)

def visualize(
    df: pd.DataFrame, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    
    for column in df.columns:
        if column != 'dim':
            # Convert boolean values to integers (0 and 1)
            y_values = df[column][1:].astype(int) if df[column].dtype == bool else df[column][1:]
            plt.plot(df['dim'][1:], y_values, label=column.capitalize())
    
    plt.xlabel('Dimension')
    plt.ylabel('Time (ms)')
    plt.title('Performance Comparison get nonzero')
    plt.legend()
    plt.grid(True)
    if save_path:
        if os.path.exists(save_path):
            os.remove(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
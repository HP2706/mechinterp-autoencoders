import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf import benchmark_decode, benchmark_models, visualize, benchmark_nonzero
from mechinterp_autoencoders import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders import TopKAutoEncoder, TopKAutoEncoderConfig
from mechinterp_autoencoders import JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig

def test():
    interval_dicts = {
        'dim': [32, 64, 128],#, 256, 512, 1024, 2048, 4096],
        'dict_mult': [64],
        'sparsity_level': [0.00001],
        'batch_size': [16],
        'use_kernel': [False],
        'use_torch_compile': [False],
        'method': ['with_loss']
    }

    df = benchmark_models(
        save_path='benchmarks/data',
        models=[
            (
                AutoEncoder, AutoEncoderConfig(dict_mult=16, d_input=768, l1_coeff=0.0001)
            ),
            (
                TopKAutoEncoder, TopKAutoEncoderConfig(dict_mult=16, d_input=768, k = 32, k_aux=32)
            ),
            (
                JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig(dict_mult=16, d_input=768, l1_coeff=0.0001)
            )
        ],
        interval_dicts=interval_dicts,
    )

    print('benchmark_models')

    benchmark_decode([32, 64, 128], save_path='benchmarks/data')
    print('benchmark_decode')

    benchmark_nonzero([32, 64, 128], save_path='benchmarks/data')
    print('benchmark_nonzero')

if __name__ == '__main__':
    test()
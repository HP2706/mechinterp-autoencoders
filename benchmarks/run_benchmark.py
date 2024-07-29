import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf import benchmark_decode, benchmark_models, visualize
from mechinterp_autoencoders import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders import TopKAutoEncoder, TopKAutoEncoderConfig
from mechinterp_autoencoders import JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig

def test():
    #df = benchmark_decode([32, 64, 128, 256, 512, 1024, 2048, 4096])
    #visualize(df, 'benchmarks/data/decode.png')

    interval_dicts = {
        'dim': [32, 64, 128], #, 256, 512, 1024, 2048, 4096],
        'dict_mult': [32],
        'sparsity_level': [0.00001],
        'batch_size': [2],
        'use_kernel': [True, False],
        'use_torch_compile': [False],
        'method': ['reconstruct']
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
        interval_dicts=interval_dicts
    )
    print(df.head())
    visualize(df, 'benchmarks/data/autoencoder.png')

if __name__ == '__main__':
    test()
from .perf import benchmark_decode, benchmark_models, visualize
from mechinterp_autoencoders import AutoEncoder, AutoEncoderConfig

def test():
    df = benchmark_decode([256, 512, 1024, 2048])
    visualize(df, 'benchmarks/data/decode.png')

    interval_dicts = {
        'dim': [32, 64, 128, 256, 512, 1024],
        'dict_mult': [16, 32, 64],
        'sparsity_level': [0.0001],
        'batch_size': [8],
        'use_kernel': [True],
        'use_torch_compile': [True]
    }

    df = benchmark_models(
        save_path='benchmarks/data',
        models=[
            (AutoEncoder, AutoEncoderConfig(dict_mult=16, d_input=768, l1_coeff=0.0001))
        ],
        interval_dicts=interval_dicts
    )
    visualize(df, 'benchmarks/data/autoencoder.png')

if __name__ == '__main__':
    test()
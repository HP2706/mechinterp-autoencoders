import pytest
import torch
from mechinterp_autoencoders.GatedAutoencoder import GatedAutoEncoderConfig, GatedAutoEncoder
from mechinterp_autoencoders.autoencoder import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders.jump_relu import JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig
from mechinterp_autoencoders.topk_autoencoder import TopKAutoEncoder, TopKAutoEncoderConfig
from mechinterp_autoencoders.utils import generate_sparse_tensor

@pytest.mark.parametrize(
    'model_cls, config_cls', 
    [
        (AutoEncoder, AutoEncoderConfig), 
        (GatedAutoEncoder, GatedAutoEncoderConfig), 
        (TopKAutoEncoder, TopKAutoEncoderConfig), 
        (JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig)
    ]
)
@pytest.mark.parametrize('dict_mult', [16, 32, 128])
@pytest.mark.parametrize('d_input', [768, 1536, 3072])
@pytest.mark.parametrize('k', [4, 8, 16])
@pytest.mark.parametrize('sparsity_level', [0.1, 0.01, 0.001])
@pytest.mark.parametrize('use_kernel', [True, False])
@pytest.mark.parametrize('batch_size', [1028, 4096, 16384])
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
        if not use_kernel:
            pytest.skip("TopKAutoEncoder requires use_kernel=True")
    
        cfg = TopKAutoEncoderConfig(
            dict_mult=dict_mult,
            d_input=d_input,
            k=k,
            k_aux=k,
            use_kernel=use_kernel
        )
    else:
        cfg = config_cls(
            dict_mult=dict_mult,
            d_input=d_input,
            use_kernel=use_kernel
        )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_cls(cfg).to(device)
    
    with torch.autocast(
        device_type=device, 
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32
    ):
        for _ in range(10):
            x = generate_sparse_tensor(batch_size, d_input, sparsity_level).to(device)
            model.forward(x, method='with_loss')

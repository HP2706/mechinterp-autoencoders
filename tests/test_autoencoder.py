import pytest
from mechinterp_autoencoders.utils import extract_nonzero, generate_sparse_tensor, get_device
import torch
from torch.nn.functional import mse_loss
from mechinterp_autoencoders.GatedAutoencoder import GatedAutoEncoderConfig, GatedAutoEncoder
from mechinterp_autoencoders.autoencoder import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders.jump_relu import JumpReLUAutoEncoder, JumpReLUAutoEncoderConfig
from mechinterp_autoencoders.topk_autoencoder import TopKAutoEncoder, TopKAutoEncoderConfig

_AutoEncoderConfig = AutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    l1_coeff=0.1,
)

_GatedAutoEncoderConfig = GatedAutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    l1_coeff=0.1,
)

_TopKAutoEncoderConfig = TopKAutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    k=1,
    k_aux=1,
    use_kernel=torch.cuda.is_available()
)

_JumpReLUConfig = JumpReLUAutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    l1_coeff=0.1,
)

@pytest.mark.parametrize(
    'model,method', [
        (AutoEncoder(cfg=_AutoEncoderConfig), 'with_loss'),
        (GatedAutoEncoder(cfg=_GatedAutoEncoderConfig), 'with_loss'),
        (TopKAutoEncoder(cfg=_TopKAutoEncoderConfig), 'with_loss'),
        (JumpReLUAutoEncoder(cfg=_JumpReLUConfig), 'with_loss'),
    ]
)
def test_forward_backward(model, method):
    if model.cfg.use_top_k:
        #we enfore sparsity by setting 90% of the values to zero
        x = torch.zeros(10, 10, device=get_device())
        indices = torch.randint(0, 10, (2,), device=get_device())
        x[:, indices] = torch.randn(indices.shape, device=get_device())
    else:
        x = torch.randn(10, 10, device=get_device())

    device = get_device()
    x = x.to(device).to(model.cfg.dtype)
    model = model.to(device)
    if isinstance(model, TopKAutoEncoder):
        out_dict = model.forward(x, ema_frequency_counter=torch.randn(model.cfg.d_sae, device=device, dtype=model.cfg.dtype), method='with_loss')
        out_dict['loss'].backward()
    else:
        out_dict = model.forward(x, method=method)
        out_dict['loss'].backward()

def test_extract_nonzero():
    x = torch.randn(1000, 10)
    values, indices = extract_nonzero(x)
    assert values.shape == indices.shape == (1000, 10)
    new_x = torch.gather(x, 1, indices)
    assert torch.allclose(new_x, values, atol=1e-5), f'new_x == values, {(new_x == values)}'

def test_generate_sparse_tensor():
    batch_size = 1024
    d_input = 768
    sparsity_level = 0.001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = generate_sparse_tensor(batch_size, d_input, sparsity_level, device)
    assert x.shape == (batch_size, d_input)
    assert torch.count_nonzero(x) == int(batch_size * d_input * sparsity_level)


def test_eager_decode():
    batch = 2
    d_in = 50
    dict_mult = 2
    d_sae = d_in * dict_mult
    k = 10
    device = get_device()

    autoencoder = TopKAutoEncoder(
        cfg=TopKAutoEncoderConfig(
            dict_mult=dict_mult, 
            d_input=d_in, 
            k=k, 
            k_aux=k, 
            use_kernel=torch.cuda.is_available()
        )
    ).to(device)

    target = torch.rand(batch, d_in, device=device)
    latents = torch.randn(batch, d_sae, device=device)
    top_vals, top_idx = latents.topk(k, dim=1)
    latents.zero_()
    latents.scatter_(1, top_idx, top_vals)

    eager_res = autoencoder.eager_decode(top_idx, top_vals)  
    standard_res = latents @ autoencoder.W_dec

    assert torch.allclose(eager_res, standard_res), f"Eager and standard results differ: {eager_res - standard_res}"
    eager_mse = mse_loss(eager_res, target).backward() # checking bwd
    standard_mse = mse_loss(standard_res, target).backward() # checking bwd


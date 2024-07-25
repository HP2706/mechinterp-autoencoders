import pytest
from mechinterp_autoencoders.GatedAutoencoder import GatedAutoEncoderConfig, GatedAutoEncoder
from mechinterp_autoencoders.autoencoder import AutoEncoder, AutoEncoderConfig
from mechinterp_autoencoders.topk_autoencoder import TopKAutoEncoder, TopKAutoEncoderConfig
import torch

_AutoEncoderConfig = AutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    l1_coeff=0.1,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    updated_anthropic_method=True,
)

_GatedAutoEncoderConfig = GatedAutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    l1_coeff=0.1,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    updated_anthropic_method=True,
)

_TopKAutoEncoderConfig = TopKAutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    k=1,
    k_aux=1,
    use_kernel=False,
)

@pytest.mark.parametrize(
    'model,method', [
        (AutoEncoder(cfg=_AutoEncoderConfig), 'with_loss'),
        (GatedAutoEncoder(cfg=_GatedAutoEncoderConfig), 'with_loss'),
        (TopKAutoEncoder(cfg=_TopKAutoEncoderConfig), 'with_loss'),
    ]
)
def test_forward(model, method):
    if isinstance(model, TopKAutoEncoder):
        model.forward(torch.randn(10, 10), ema_frequency_counter=torch.randn(model.d_hidden), method='with_loss')
    else:
        model.forward(torch.randn(10, 10), method=method)

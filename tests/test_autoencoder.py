import pytest
from mechinterp_autoencoders.autoencoder import AutoEncoder, AutoEncoderConfig
import torch

_AutoEncoderConfig = AutoEncoderConfig(
    dict_mult=1,
    d_input=10,
    l1_coeff=0.1,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    updated_anthropic_method=True,
)

@pytest.mark.parametrize(
    'model,method', [
        (AutoEncoder(cfg=_AutoEncoderConfig), 'with_loss'),
    ]
)
def test_forward(model, method):

    model.forward(torch.randn(10, 10), method=method)

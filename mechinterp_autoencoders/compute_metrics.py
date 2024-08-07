from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
import torch

def did_fire(acts : Float[Tensor, "batch_size dim"])-> Tensor:
    '''compute count of how often a feature was nonzero across the batch'''
    return (acts > 0).long().sum(dim=0)

def avg_num_firing_per_neuron(x: Float[Tensor, "batch_size dim"]) -> Tensor:
    '''compute average number of times a neuron fires across the batch'''
    return (x > 0).float().sum(dim=1).mean()

def l0_norm(x: Float[Tensor, "batch_size dim"]) -> Tensor:
    '''the mean l0 norm of the activations over the batch'''
    return x.ne(0).float().sum(dim=1).mean(dim=0)


def normalized_mse(x: Float[Tensor, "batch_size dim"], ground_truth: Float[Tensor, "batch_size dim"]) -> Tensor:
    return (
        (x - ground_truth).pow(2).mean(dim=1) / ((ground_truth**2) + 1e-5).mean(dim=1) #we use 1e-5 to avoid division by zero 
    ).mean()

def mean_absolute_error(x: Float[Tensor, "batch_size dim"], ground_truth: Float[Tensor, "batch_size dim"]) -> Tensor:
    return (torch.abs(x - ground_truth)).mean()

def l1_norm(x: Float[Tensor, "batch_size dim"]) -> Tensor:
    '''the mean l1 norm of the activations over the batch'''
    return x.abs().sum(1).mean(0)

def normalized_L1_loss(
    latent_activations: Float[Tensor, "batch_size d_sae"],
    ground_truth: Float[Tensor, "batch_size dim"],
) -> Tensor:
    '''the mean normalized L1 loss of the activations over the batch'''
    return (latent_activations.abs().sum(dim=1) / ground_truth.norm(dim=1)).mean()

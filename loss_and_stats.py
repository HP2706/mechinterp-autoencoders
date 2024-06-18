from torch import Tensor
import torch

def compute_did_fire(acts : Tensor)-> Tensor:
    '''compute count of how often specific feature was nonzero across the batch'''
    return (acts > 0).long().sum(dim=0)

def compute_avg_num_firing_per_neuron(x: Tensor) -> Tensor:
    #compute average number of times a neuron fires across the batch
    return (x > 0).float().sum(dim=1).mean()

def compute_l0_norm(x: Tensor) -> Tensor:
    '''the mean l0 norm of the activations over the batch'''
    #x : shape(batch_size, dim)
    assert len(x.shape) == 2
    nonzero_per_activation = x.ne(0).float().sum(dim=1)
    return nonzero_per_activation.mean(0)

def compute_normalized_mse(x: Tensor, ground_truth: Tensor) -> Tensor:
    return (
        (x - ground_truth).pow(2).mean(dim=1) / (ground_truth**2).mean(dim=1)
    ).mean()

def compute_mean_absolute_error(x: Tensor, ground_truth: Tensor) -> Tensor:
    return (torch.abs(x - ground_truth)).mean()

def compute_l1_sparsity(x: Tensor) -> Tensor:
    return x.abs().sum(1).mean(0)

def compute_normalized_L1_loss(
    latent_activations: Tensor,
    ground_truth: Tensor,
) -> Tensor:
    return (latent_activations.abs().sum(dim=1) / ground_truth.norm(dim=1)).mean()

from torch import Tensor
import torch

def compute_did_fire(acts : Tensor)-> Tensor:
    '''compute count of how often specific feature was nonzero across the batch'''
    return (acts > 0).long().sum(dim=0)

def compute_mean_firing_percentage(x : Tensor)-> Tensor:
    '''compute the mean firing percentage of the neurons across the batch'''
    return (x > 0).float().mean(dim=(0,1))

def compute_mean_firing_per_batch(x : Tensor)-> Tensor:
    '''compute the mean firing percentage of the neurons across the batch'''
    return (x > 0).float().mean(dim=(0,1))

def compute_avg_num_firing_per_neuron(x: Tensor) -> Tensor:
    #compute average number of times a neuron fires across the batch
    return (x > 0).float().sum(dim=1)


def compute_l0_norm(x: Tensor) -> Tensor:
    '''the mean l0 norm of the activations over the batch'''
    #x : shape(batch_size, dim)
    assert len(x.shape) == 2
    nonzero_per_activation = x.ne(0).float().sum(dim=1)
    return nonzero_per_activation.mean(0)

def compute_mean_absolute_error(x: Tensor, ground_truth: Tensor) -> Tensor:
    return (torch.abs(x - ground_truth)).mean()

def compute_mse(x: Tensor, ground_truth: Tensor) -> Tensor:
    return (x - ground_truth).pow(2).mean()

def compute_l1_sparsity(x: Tensor) -> Tensor:
    return x.abs().sum(1).mean(0)

def compute_normalized_mse(x : Tensor, ground_truth : Tensor) -> Tensor:
    return ((x - ground_truth) ** 2).mean(dim=1) / (ground_truth**2).mean(dim=1)

def compute_normalized_L1_loss(
    latent_activations: Tensor,
    ground_truth: Tensor,
) -> Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / ground_truth.norm(dim=1)).mean()

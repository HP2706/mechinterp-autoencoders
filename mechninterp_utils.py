from _types import Methods
import pandas as pd
from autoencoder import compute_mse
import torch
from tqdm import tqdm
import math
from functools import partial
from transformer_lens import utils
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Union, Optional
from autoencoder import AutoEncoder, GatedAutoEncoder
from torchmetrics.regression import SpearmanCorrCoef
import numpy as np
import plotly.express as px
from _types import Loss_Method
from torch.distributions import Categorical
from datamodels import AnthropicResample

#code for plotting histogram taken from https://github.com/ArthurConmy/sae/tree/8bf510d9285eb5d79f77fe6896f2166d35f06a2b
# Define a set of arguments which are passed to fig.update_layout (rather than just being included in e.g. px.imshow)
UPDATE_LAYOUT_SET = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_type", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "font",
    "modebar_add", "legend_traceorder", "autosize", "coloraxis_colorbar_tickformat", "font_family", "font_size",
}
# Gives options to draw on plots, and remove plotly logo
CONFIG = {'displaylogo': False}
CONFIG_STATIC = {'displaylogo': False, 'staticPlot': True}
MODEBAR_ADD = ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def hist(tensor : torch.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in UPDATE_LAYOUT_SET}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in UPDATE_LAYOUT_SET}
    draw = kwargs_pre.pop("draw", True)
    static = kwargs_pre.pop("static", False)
    return_fig = kwargs_pre.pop("return_fig", False)


    if "modebar_add" not in kwargs_post:
        kwargs_post["modebar_add"] = ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
    add_mean_line = kwargs_pre.pop("add_mean_line", False)
    names = kwargs_pre.pop("names", None)
    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "autosize" not in kwargs_post:
        kwargs_post["autosize"] = False
    # print(kwargs_pre, "and", kwargs_post)
    arr = to_numpy(tensor)
    fig = px.histogram(x=arr, **kwargs_pre)
    fig.update_layout(**kwargs_post)
    if add_mean_line:
        if arr.ndim == 1:
            fig.add_vline(x=arr.mean(), line_width=3, line_dash="dash", line_color="black", annotation_text=f"Mean = {arr.mean():.3f}", annotation_position="top")
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                fig.add_vline(x=arr[i].mean(), line_width=3, line_dash="dash", line_color="black", annotation_text=f"Mean = {arr.mean():.3f}", annotation_position="top")
    if names is not None:
        for i in range(len(fig.data)): # type: ignore
            fig.data[i]["name"] = names[i // 2 if "marginal" in kwargs_pre else i] #type: ignore
    if draw: 
        fig.update_layout(modebar_add=MODEBAR_ADD)
    else:
        fig.update_layout(modebar_add=[])
    if return_fig:
        return fig
    else:
        fig.show(renderer=renderer, config=CONFIG_STATIC if static else CONFIG)



def scale_dataset(X: torch.Tensor, n: float):
    '''Computes the expected norm of the dataset row (dim=-1) and normalizes to sqrt(target_norm).'''
    n_sqrt = math.sqrt(n)
    norms = torch.norm(X, dim=-1, p=2)  # Compute L2 norm of each row
    mean_norm = torch.mean(norms).float()
    scaling_factor = n_sqrt / mean_norm
    X_scaled = X * scaling_factor  # Scale the dataset
    return X_scaled

@torch.no_grad()
def anthropic_resample(
    indices: torch.Tensor,
    val_dataset: DataLoader,
    model: Union[AutoEncoder, GatedAutoEncoder],
    optimizer: torch.optim.Optimizer,
    resampling_dataset_size: int,
    sched: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    resample_factor: float = 0.2,
    bias_resample_factor: float = 0.2,
) -> AnthropicResample:
    '''
    An implementation of the anthropic resampling algorithm.
    The algorithm is as follows:
    1.At training steps 25,000, 50,000, 75,000 and 100,000, identify which neurons have not fired in any of the previous 12,500 training steps.
    2.Compute the loss for the current model on a random subset of 819,200 inputs.
    3.Assign each input vector a probability of being picked that is proportional to the square of the autoencoder’s loss on that input.
    4.For each dead neuron sample an input according to these probabilities. Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector for the dead autoencoder neuron.
    5.For the corresponding encoder vector, renormalize the input vector to equal the average norm of the encoder weights for alive neurons × 0.2. Set the corresponding encoder bias element to zero.
    6.Reset the Adam optimizer parameters for every modified weight and bias term.
    '''
    assert isinstance(val_dataset, DataLoader)
    assert val_dataset.batch_size is not None

    global_loss_increases = torch.zeros((resampling_dataset_size,), dtype=model.cfg.dtype, device=model.cfg.device)
    d_in = model.W_enc.shape[0]
    global_input_activations = torch.zeros((resampling_dataset_size, d_in), dtype=model.cfg.dtype, device=model.cfg.device)

    for (batch_idx, normal_activations) in enumerate(val_dataset): # , total=resampling_dataset_size):
        if batch_idx * val_dataset.batch_size >= resampling_dataset_size:
            break

        normal_activations = normal_activations.to(model.cfg.device)
        x_reconstruct = model.forward(normal_activations, method='reconstruct') 
        loss = (x_reconstruct - normal_activations).pow(2).sum(1)  # Sum over features to get loss per sample
        changes_in_loss_dist = Categorical(
            torch.nn.functional.relu(loss) / torch.nn.functional.relu(loss).sum()
        )

        samples = changes_in_loss_dist.sample((val_dataset.batch_size,)) #type: ignore
        assert samples.shape == (val_dataset.batch_size,), f"{samples.shape=}; {val_dataset.batch_size=}"
        batch_idx = batch_idx * val_dataset.batch_size #type: ignore
        global_loss_increases[
            batch_idx: batch_idx + val_dataset.batch_size
        ] = loss[samples].cpu()
        global_input_activations[
            batch_idx: batch_idx + val_dataset.batch_size
        ] = normal_activations[samples].cpu()

    sample_indices = torch.multinomial(
        global_loss_increases / global_loss_increases.sum(),
        len(indices), 
        replacement=False,
    )

    model.W_dec.data[indices, :] = (
        (
            global_input_activations[sample_indices]
            / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
        )
        .to(model.cfg.dtype)
        .to(model.device)
    )

    # Set W_enc equal to W_dec.T in these indices, first
    model.W_enc.data[:, indices] = model.W_dec.data[indices, :].T
    if indices.sum() < model.d_sae:
        all_indices = torch.arange(model.d_sae, device=indices.device)
        alive_indices = all_indices[~torch.isin(all_indices, indices)]
        average_alive_norm = torch.norm(model.W_enc.data[alive_indices, :], dim=0).mean() 
        # we compute mean norm of alive features

        model.W_enc.data[:, indices] *= resample_factor * average_alive_norm
        relevant_biases = model.b_enc.data[indices].mean()

        # Set biases to resampled value
        model.b_enc.data[indices] = relevant_biases * bias_resample_factor

        out = AnthropicResample(
            inactive_features=indices,
            resample_norm=average_alive_norm.item()
        )

    else:
        model.W_enc.data[:, indices] *= resample_factor
        model.b_enc.data[indices] = - 5.0
        out = AnthropicResample(
            inactive_features=indices,
            resample_norm=torch.norm(model.W_enc.data, dim=0).mean().item() #average norm
        )

    # Reset the Adam optimizer parameters for every modified weight and bias term
    #taken from https://github.com/ArthurConmy/sae
    indices = indices.to(model.device)
    #TODO this does not work for GATEDAUTOENCODER!!
    
    model.zero_optim_grads(optimizer=optimizer, indices=indices)

    print("checking everything is set correctly")
    # Check that the opt is really updated
    for dict_idx, (k, v) in enumerate(optimizer.state.items()):
        for v_key in ["exp_avg", "exp_avg_sq"]:
            if dict_idx == 0:
                if k.data.shape != (model.d_sae, model.d_in):
                    print(
                        "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                    )
                if v[v_key][indices, :].abs().max().item() > 1e-6:
                    print(
                        "Warning: it does not seem as if resetting the Adam parameters worked"
                    )

    if sched is not None:
        # Keep on stepping till we're cfg["lr"] * cfg["sched_lr_factor"]
        max_iters = 10**7
        while sched.get_last_lr()[0] > model.metadata_cfg.lr * model.metadata_cfg.sched_lr_factor + 1e-9: #type: ignore
            sched.step()
            max_iters -= 1
            if max_iters == 0:
                raise ValueError("Too many iterations -- sched is messed up")

        print("sched final lr", sched.get_last_lr())
    return out

#for antrhopic interpretability paper
def torch_spearman_correlation(
    predicted : torch.Tensor, 
    actual : torch.Tensor
) -> torch.Tensor:
    '''computes the spearman correlation between predicted and actual activations.
    From the antrhopic auto encoder paper
    Args:
        predicted : torch.Tensor of shape [batch=60] 
        actual : torch.Tensor of shape [batch=60]
    '''
    spearman = SpearmanCorrCoef()
    return spearman(predicted, actual) 

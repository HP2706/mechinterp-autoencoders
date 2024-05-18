from _types import Methods
import pandas as pd
from autoencoder import compute_mse
from torch.distributions.multinomial import Categorical
import torch
from tqdm import tqdm
import math
from functools import partial
from transformer_lens import utils
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Union
from autoencoder import AutoEncoder, GatedAutoEncoder
from torchmetrics.regression import SpearmanCorrCoef
import numpy as np
import plotly.express as px
from _types import Loss_Method

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

def anthropic_resample(
    indices: torch.Tensor,
    val_dataset: Dataset,
    model: Union[AutoEncoder, GatedAutoEncoder],
    optimizer: torch.optim.Optimizer,
    anthropic_resample_batches: int = 100,
    batch_size: int = 4096,
    sample_size: int = 819200,
    resample_factor: float = 0.2,
):
    
    anthropic_iterator = range(0, anthropic_resample_batches, batch_size)
    anthropic_iterator = tqdm(anthropic_iterator, desc="Anthropic loss calculating")
    global_loss_increases = torch.zeros((sample_size,), dtype=model.cfg.dtype, device=model.cfg.device)
    d_in = model.W_enc.shape[0]
    global_input_activations = torch.zeros((sample_size, d_in), dtype=model.cfg.dtype, device=model.cfg.device)

    for batch_idx in anthropic_iterator:
        normal_activations: torch.Tensor = val_dataset[batch_size:batch_idx+batch_size]
        print("batch", normal_activations.shape)
        x_reconstruct = model.forward(normal_activations, method='with_loss').x_reconstruct 
        loss = (x_reconstruct - normal_activations).pow(2) # we don't take the mean 
        changes_in_loss_dist = Categorical(
            torch.nn.functional.relu(loss) / torch.nn.functional.relu(loss).sum(dim=1, keepdim=True)
        )

        samples = changes_in_loss_dist.sample()
        assert samples.shape == (batch_size,), f"{samples.shape=}; {batch_size=}"

        global_loss_increases[
            batch_idx: batch_idx + batch_size
        ] = loss[torch.arange(batch_size), samples]
        global_input_activations[
            batch_idx: batch_idx + batch_size
        ] = normal_activations[torch.arange(batch_size), samples]

    sample_indices = torch.multinomial(
        global_loss_increases / global_loss_increases.sum(),
        len(indices), 
        replacement=False,
    )

    # Replace W_dec with normalized versions of these
    model.W_dec.data[indices, :] = (
        (
            global_input_activations[sample_indices]
            / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
        )
        .to(model.dtype)
        .to(model.device)
    )

    # Set W_enc equal to W_dec.T in these indices, first
    model.W_enc.data[:, indices] = model.W_dec.data[indices, :].T

    # Renormalize the encoder vector
    alive_neurons = indices[indices > 0]
    mean_norm = torch.mean(torch.norm(model.W_enc.data[:, alive_neurons], dim=0))
    model.W_enc.data[:, indices] *= (resample_factor * mean_norm / torch.norm(model.W_enc.data[:, indices], dim=0))

    # Set the corresponding encoder bias element to zero
    model.b_enc.data[indices] = 0.0

    # Reset the Adam optimizer parameters for every modified weight and bias term
    for param in [model.W_dec, model.W_enc, model.b_enc]:
        if param in optimizer.state:
            del optimizer.state[param]


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

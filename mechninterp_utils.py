import torch
from functools import partial
from transformer_lens import utils
from typing import Union
from autoencoder import AutoEncoder, GatedAutoEncoder
from torchmetrics.regression import SpearmanCorrCoef
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

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

def anthropic_resample(
    self,
    indices,
    model : Union[AutoEncoder, GatedAutoEncoder]
):
    #TODO implement this 
    
    anthropic_iterator = range(0, self.cfg["anthropic_resample_batches"], self.cfg["batch_size"])
    total_size = len(anthropic_iterator) * self.cfg["batch_size"] * self.cfg["seq_len"]
    anthropic_iterator = tqdm(anthropic_iterator, desc="Anthropic loss calculating")
    global_loss_increases = torch.zeros((self.cfg["anthropic_resample_batches"],), dtype=self.dtype, device=self.device)
    global_input_activations = torch.zeros((self.cfg["anthropic_resample_batches"], self.d_in), dtype=self.dtype, device=self.device)

    for refill_batch_idx_start in anthropic_iterator:
        #TODO get test set


        # Do a forwards pass, including calculating loss increase

        normal_loss = None

        normal_loss = normal_loss.cpu()
        changes_in_loss = sae_loss - normal_loss
        changes_in_loss_dist = Categorical(
            torch.nn.functional.relu(changes_in_loss) / torch.nn.functional.relu(changes_in_loss).sum(dim=1, keepdim=True)
        )
        samples = changes_in_loss_dist.sample()
        assert samples.shape == (self.cfg["batch_size"],), f"{samples.shape=}; {self.cfg['batch_size']=}"
        
        global_loss_increases[
            refill_batch_idx_start: refill_batch_idx_start + self.cfg["batch_size"]
        ] = changes_in_loss[torch.arange(self.cfg["batch_size"]), samples]
        global_input_activations[
            refill_batch_idx_start: refill_batch_idx_start + self.cfg["batch_size"]
        ] = normal_activations[torch.arange(self.cfg["batch_size"]), samples]

    sample_indices = torch.multinomial(
        global_loss_increases / global_loss_increases.sum(),
        len(indices), 
        replacement=False,
    )

    # Replace W_dec with normalized versions of these
    self.W_dec.data[indices, :] = (
        (
            global_input_activations[sample_indices]
            / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
        )
        .to(self.dtype)
        .to(self.device)
    )

    # Set W_enc equal to W_dec.T in these indices, first
    self.W_enc.data[:, indices] = self.W_dec.data[indices, :].T

    # Then, change norms to be equal to a factor (0.2 in Anthropic) times the average norm of all the other columns, if other columns exist
    if indices.shape[0] < self.d_sae:
        sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
        sum_of_all_norms -= len(indices)
        average_norm = sum_of_all_norms / (self.d_sae - len(indices))
        metrics["resample_norm_thats_hopefully_less_or_around_one"] = average_norm.item()
        self.W_enc.data[:, indices] *= self.cfg["resample_factor"] * average_norm

        # Set biases to resampledvalue
        relevant_biases = self.b_enc.data[indices].mean()
        self.b_enc.data[indices] = relevant_biases * self.cfg["bias_resample_factor"]

    else:
        self.W_enc.data[:, indices] *= self.cfg["resample_factor"]

        self.b_enc.data[indices] = - 5.0

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

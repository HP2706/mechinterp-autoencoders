import tqdm
import torch
from jaxtyping import Float
from transformer_lens import utils
from functools import partial


def compute_proxy(proba_distb : torch.Tensor, feature_tokens : torch.Tensor): #NOT finished
    # shape of proba_distb is [batch, seq_len, vocab_size]
    # shape of feature_tokens is [n_features]
    '''assumes we get a full probability distribution over the vocabulary'''
    assert proba_distb.sum(dim=-1) - 1 < 1e-5
    #compute p(s)
    p_s = torch.prod(proba_distb, dim=-1)  # [batch, seq_len]
    #compute p(s|feature)
    p_feature_given_s = proba_distb.gather(dim=-1, index=feature_tokens.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len, n_features]
    p_feature_given_s = p_feature_given_s.prod(dim=-1)  # [batch, seq_len]
    return p_s, p_feature_given_s  # [batch, seq_len]

""" 
from torchmetrics.regression import SpearmanCorrCoef
def torch_spearman_correlation(
        predicted : Float[torch.Tensor, "batch token_activations"], 
        actual : Float[torch.Tensor, "batch token_activations"]
    ) -> Float[torch.Tensor, "batch"]:
    '''computes the spearman correlation between predicted and actual activations.
    From the antrhopic auto encoder paper
    Args:
        predicted : torch.Tensor of shape [batch=60, token_activations=9] 
        actual : torch.Tensor of shape [batch=60, token_activations=9]
    '''
    spearman = SpearmanCorrCoef(num_outputs=actual.shape[1])
    return spearman(predicted, actual) 
"""

def replacement_hook(mlp_post : torch.Tensor, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post : torch.Tensor, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post : torch.Tensor, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(
    model, 
    encoder, 
    all_tokens : torch.Tensor,
    model_batch_size: int,
    num_batches : int = 5
):
    
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:model_batch_size]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replacement_hook, encoder=encoder))]
        )
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), zero_ablate_hook)]
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"Reconstruction Score: {score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss



@torch.no_grad()
def get_freqs(
    tokens,
    encoder, 
    model,
    num_batches=25,
    device="cuda"
):
    
    act_freq_scores = torch.zeros(encoder.d_hidden, dtype=torch.float32).to(device)
    total = 0
    for i in tqdm.trange(num_batches):

        _, cache = model.run_with_cache(
            tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0)
        )
        mlp_acts = cache[utils.get_act_name("post", 0)]
        mlp_acts = mlp_acts.reshape(-1, d_mlp)

        hidden = encoder(mlp_acts)[2]

        act_freq_scores += (hidden > 0).sum(0) 
        #increments across the sequence dimension 
        # each feature is incremented if its value is greater than 0
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores
import torch
from functools import partial
from transformer_lens import utils
from torchmetrics.regression import SpearmanCorrCoef
from jaxtyping import Float

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

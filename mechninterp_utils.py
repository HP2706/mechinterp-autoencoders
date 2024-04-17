import torch

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


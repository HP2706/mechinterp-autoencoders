from gguf import Any
import tqdm
import torch
from typing import List, Tuple, Optional
from torch.nn import functional as F
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset, Dataset
from autoencoder import AutoEncoder
from pydantic import BaseModel, field_validator
import plotly.express as px


class PipelineConfig(BaseModel):
    device: str
    d_type: Any = torch.float32
    batch_size: int = 512

    @field_validator('d_type')
    @classmethod
    def check_d_type(cls, v):
        if v not in [torch.float32, torch.float16]:
            raise ValueError("d_type must be 'fp32' or 'fp16'")
        return v

class MechInterpPipeline:
    def __init__(
        self,
        model_name : str,
        encoder_name : str,
        dataset_name : str,
        cfg : PipelineConfig
    ) -> None:

        model = HookedTransformer.from_pretrained(model_name, device=cfg.device).to(cfg.d_type)

        self.model = model
        self.encoder = AutoEncoder.load_from_hf(encoder_name, cfg.device)
        data : Dataset = load_dataset(dataset_name, split="train") # type: ignore
        tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
        self.all_tokens : torch.Tensor = tokenized_data["tokens"] # type: ignore
        self.cfg = cfg

    @torch.no_grad()
    def get_freqs(
        self,
        tokens, 
        d_mlp : int,
        num_batches=25,
        device="cuda"
    ):
        
        act_freq_scores = torch.zeros(self.encoder.d_hidden, dtype=torch.float32).to(device)
        total = 0
        for i in tqdm.trange(num_batches):

            _, cache = self.model.run_with_cache(
                tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0)
            )
            mlp_acts = cache[utils.get_act_name("post", 0)]
            mlp_acts = mlp_acts.reshape(-1, d_mlp)

            hidden = self.encoder(mlp_acts)[2]

            act_freq_scores += (hidden > 0).sum(0) 
            #increments across the sequence dimension 
            # each feature is incremented if its value is greater than 0
            total+=hidden.shape[0]
        act_freq_scores /= total
        num_dead = (act_freq_scores==0).float().mean()
        print("Num dead", num_dead)
        return act_freq_scores.cpu()

    def get_examples(self, batch_size : int, seq_len : int):
        '''gets examples for individual feature with seq'''
        tokens = self.all_tokens[torch.randperm(len(self.all_tokens))[:self.cfg.batch_size]]
        max_seq_len = tokens.shape[1]
        start_index = torch.randint(0, max_seq_len - seq_len + 1, (1,)).item()

        # Slice the tokens to get a continuous chunk of length seq_len
        tokens = tokens[:, start_index:start_index + seq_len]
        tokens = tokens[:batch_size, :seq_len]
        return tokens

    def tokens_to_str(
        self,
        batch_tokens : torch.Tensor
    )-> List[str]: #batch of strings
        #batch_tokens shape is (batch_size, seq_len) or (seq_len)
        if len(batch_tokens.shape) == 1:
            batch_tokens = batch_tokens.unsqueeze(0)
        return [''.join(self.model.to_str_tokens(tokens)) for tokens in batch_tokens] #type: ignore

    @torch.no_grad()
    def get_feature_acts(
        self,     
        tokens : torch.Tensor, 
        feature_index : int
    ) -> torch.Tensor:
        '''gets the activation values for a feature index'''
        _, cache = self.model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)] # shape (batch, seq_len, d_mlp)
        feature_in = self.encoder.W_enc[:, feature_index]
        feature_bias = self.encoder.b_enc[feature_index]
        feature_acts = F.relu((mlp_acts - self.encoder.b_dec) @ feature_in + feature_bias) # shape (batch, seq_len)
        return feature_acts.cpu()

    @torch.no_grad()
    def match_activations_tokens(
        self, 
        tokens : torch.Tensor,
        feature_index : int,
    ) -> List[tuple[str, List[tuple[int, str, float]]]]:
        
        activations = self.get_feature_acts(tokens, feature_index)

        batch_interp = []
        for batch_idx in range(len(activations)):
            non_zero_indices = torch.nonzero(activations[batch_idx], as_tuple=True)
            token_activation_pairs = []
            if non_zero_indices[0].nelement() == 0: # if all are zeros we skip
                continue
            else:
                for tensor in non_zero_indices:
                    print(tensor)
                    all_indices = tensor.tolist()
                    print(all_indices)
                    for active_idx in all_indices:
                        active_tokens = tokens[batch_idx,active_idx]
                        non_zero_values = activations[batch_idx,active_idx]

                        token_activation_pairs.append((active_idx , self.model.to_string(active_tokens), non_zero_values))
                text = self.tokens_to_str(tokens[batch_idx])
            if token_activation_pairs != []:
                batch_interp.append((text, token_activation_pairs)) 
        return batch_interp

    def filter_non_zero(
        self,
        tokens : torch.Tensor,
        activations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''filters out the non-zero activations
        activations shape is (batch_size, seq_len) or (seq_len)
        Args:
            tokens: (batch_size, seq_len) or (seq_len)
            activations: (batch_size, seq_len) or (seq_len)
        Returns:
            non_zero_activations
            non_zero_indices
            non_zero_tokens
        '''
        non_zero_indices = torch.nonzero(activations, as_tuple=True)
        non_zero_activations = activations[non_zero_indices]
        non_zero_tokens = tokens[non_zero_indices]
        if len(non_zero_tokens.shape) == 1:
            non_zero_indices = non_zero_indices[0]
        else:
            non_zero_indices = torch.vstack(non_zero_indices)

        return non_zero_activations, non_zero_indices, non_zero_tokens


    def get_batch_w_repeated_tokens(
        self,
        all_tokens: torch.Tensor, 
        target_tokens: List[int], 
        target_count: Optional[int] = None
    ) -> torch.Tensor:
        """
        Gets a batch of tokens where each sequence contains all tokens specified in target_tokens.
        """
        # Create a mask where each sequence that contains all target_tokens is True
        contains_targets = torch.stack([all_tokens == target_token for target_token in target_tokens]).all(dim=0).any(dim=1)
        # Filter sequences that contain all target_tokens
        filtered_sequences = all_tokens[contains_targets]
        
        # Limit the number of sequences to target_count
        if target_count is not None and len(filtered_sequences) > target_count:
            filtered_sequences = filtered_sequences[:target_count]
        
        return filtered_sequences

    def get_activating_tokens_sorted(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,        
    ):
        #we remove zeros
        non_zero_activations, idxs, flatten_tokens = self.filter_non_zero(tokens, activations)
        sorted_idxs = torch.argsort(non_zero_activations, descending=True)
        sorted_activations = non_zero_activations[sorted_idxs]
        sorted_tokens = flatten_tokens[sorted_idxs]

        sorted_tokens = [self.model.tokenizer.decode(token_id) for token_id in  sorted_tokens[sorted_idxs]]
        fig = px.bar(x=sorted_tokens, y=sorted_activations)
        return fig
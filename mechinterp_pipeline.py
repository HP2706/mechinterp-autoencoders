from multiprocessing import context
import tqdm
import torch
from typing import List, Tuple, Optional, Any, Union
from torch.nn import functional as F
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset, Dataset
from autoencoder import AutoEncoder
from pydantic import BaseModel, field_validator
import plotly.express as px
from datamodels import ActivationExample


FullData = List[tuple[str, List[ActivationExample]]]
OnlyActivations = List[ActivationExample]


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
        self.encoder = AutoEncoder.load_from_hf(encoder_name, cfg.device).to(cfg.device)
        data : Dataset = load_dataset(dataset_name, split="train")# type: ignore
        data = data.select(range(int(0.1*len(data))))
        tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
        self.all_tokens : torch.Tensor = tokenized_data["tokens"] # type: ignore
        self.cfg = cfg

    def get_tokens(
        self,
        target_tokens: Optional[List[int]] = None, 
        shuffle: bool = True,
        batch_size: int = 100,
        seq_len: int = 100,
    ) -> torch.Tensor:
        """
        Gets a batch of tokens where each sequence contains all tokens specified in target_tokens.
        """
        if seq_len > self.all_tokens.shape[1]:
            raise ValueError(f"Sequence length {seq_len} cannot be longer than the maximum sequence length in the dataset {self.all_tokens.shape[1]}")

        if shuffle:
            tokens = self.all_tokens[torch.randperm(len(self.all_tokens))[:self.cfg.batch_size]]
        else:
            tokens = self.all_tokens

        # Create a mask where each sequence that contains all target_tokens is True
        if target_tokens:
            contains_targets = torch.stack([tokens == target_token for target_token in target_tokens]).all(dim=0).any(dim=1)
            # Filter sequences that contain all target_tokens
            tokens = tokens[contains_targets]
        if len(tokens) > batch_size:
            tokens = tokens[:batch_size, :seq_len]
        
        return tokens

    @torch.no_grad()
    def get_freqs(
        self,
        tokens, 
        d_mlp : int,
        num_batches=25,
    ) -> torch.Tensor:
        
        act_freq_scores = torch.zeros(self.encoder.d_hidden, dtype=torch.float32).to(self.cfg.device)
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

        if tokens.device != self.cfg.device:
            tokens = tokens.to(self.cfg.device)

        _, cache = self.model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)] # shape (batch, seq_len, d_mlp)
        feature_in = self.encoder.W_enc[:, feature_index]
        feature_bias = self.encoder.b_enc[feature_index]
        feature_acts = F.relu((mlp_acts - self.encoder.b_dec) @ feature_in + feature_bias) # shape (batch, seq_len)
        return feature_acts.cpu()
        
    @torch.no_grad()
    def get_non_threshold_activations_with_tokens(
        self, 
        feature_index: int,
        context_window: int = 4, # on both sides of token from anthropic paper
        tokens: Optional[torch.Tensor] = None,
        threshold: float = 0.01,
        entire_dataset: bool = False,
        with_entire_context: bool = False
    ) -> Union[FullData, OnlyActivations]:
        '''gets the non zero activations for a feature index and the corresponding text
        Args:
            tokens: (batch_size, seq_len) or (seq_len)
            feature_index: int
        Returns:
            List[tuple[str, List[tuple[int, str, float]]]]: List of len batch_size of
            tuples where each tuple contains the text in the batch and the list of activation pairs for the feature index
        '''
        
        # Check inputs
        if tokens is not None and entire_dataset:
            raise ValueError("Cannot specify both tokens and entire_dataset, to use entire dataset, set tokens to None")

        batch_interp = []

        if tokens is None:
            if entire_dataset:
                print("Getting activations for entire dataset")
                dataset_size = len(self.all_tokens)
                print("Dataset size", dataset_size)

                for batch_idx in tqdm.tqdm(range(0, dataset_size, self.cfg.batch_size)):
                    batch_tokens = self.all_tokens[batch_idx:batch_idx + self.cfg.batch_size]
                    batch_activations = self.get_feature_acts(batch_tokens, feature_index)
                    batch_interp.extend(
                        self.process_activations(
                            batch_tokens, batch_activations, threshold, context_window, with_entire_context
                        )
                    )
            else:
                tokens = self.get_tokens(
                    target_tokens = [feature_index], 
                    batch_size = self.cfg.batch_size
                )
                activations = self.get_feature_acts(tokens, feature_index)
                batch_interp = self.process_activations(
                    tokens, activations, threshold, context_window, with_entire_context
                )
        else:
            activations = self.get_feature_acts(tokens, feature_index)
            batch_interp = self.process_activations(
                tokens, activations, threshold, context_window, with_entire_context
            )

        return batch_interp

    def process_activations(
        self, 
        tokens, 
        activations, 
        threshold, 
        context_window,
        with_entire_context: bool
    ) -> Union[FullData, OnlyActivations]:
        batch_results = []
        for batch_idx in range(len(activations)):
            batch_tokens = tokens[batch_idx]
            non_zero_acts, non_zero_tokens = self.filter_non_zero_sequence(
                batch_tokens, activations[batch_idx], threshold
            ) #TODO fix the context problem, it doesnt produce 9 tokens each time.

            token_activation_pairs = []
            if non_zero_acts.numel() == 0:  # no non-zero activations, we skip
                continue

            token_activation_pairs = []
            texts : List[str] = self.model.tokenizer.batch_decode(non_zero_tokens)
            for idx, (token_str, act) in enumerate(zip(texts, non_zero_acts.tolist())):


                start = max(0, idx - context_window)
                end = min(len(batch_tokens), idx + context_window + 1)
                
                if abs(idx - start) < context_window:
                    diff = abs(idx - start)
                    end += context_window - diff
                elif abs(idx - end) < context_window:
                    diff = abs(idx - end)
                    start -= context_window - diff
                
                start = max(0, start)
                end = min(len(batch_tokens), end)
                #part1 = self.model.tokenizer.batch_decode(batch_tokens[start:idx])
                #part2 = self.model.tokenizer.batch_decode(batch_tokens[idx:end])
                #context = part1 + [f"|{token_str}|"] + part2
                context = self.model.tokenizer.batch_decode(batch_tokens[start:end])
                #print("difference", end - start, "idx", idx, "n tokens sequence", len(batch_tokens), "text", ''.join(context) )

                token_activation_pairs.append(
                    ActivationExample(
                        text=token_str, 
                        token_id=idx, 
                        activation=act,
                        context= ''.join(context)
                    )
                )
            if with_entire_context:
                text = self.tokens_to_str(tokens[batch_idx])  # we save the entire text
                batch_results.append((text, token_activation_pairs))
            else:
                batch_results.append(token_activation_pairs)
        return batch_results


    def filter_non_zero_sequence(
        self,
        tokens : torch.Tensor,
        activations: torch.Tensor,
        threshold : Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''filters out the non-zero activations
        activations shape is (batch_size, seq_len) or (seq_len)
        Args:
            tokens: (batch_size, seq_len) or (seq_len)
            activations: (batch_size, seq_len) or (seq_len)
        Returns:
            non_zero_activations : (n_non_zero_activations)
            non_zero_tokens : (n_non_zero_activations)
        '''
        non_zero_indices = torch.nonzero(activations, as_tuple=True)
        non_zero_activations = activations[non_zero_indices]
        non_zero_tokens = tokens[non_zero_indices]
        if threshold is not None:
            condition = non_zero_activations > threshold
            non_zero_activations = non_zero_activations[condition]
            non_zero_tokens = non_zero_tokens[condition]
        return non_zero_activations, non_zero_tokens

    def get_activating_tokens_sorted(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,        
    ):
        #we remove zeros
        non_zero_activations, flatten_tokens = self.filter_non_zero_sequence(tokens, activations)
        sorted_idxs = torch.argsort(non_zero_activations, descending=True)
        sorted_activations = non_zero_activations[sorted_idxs]
        sorted_tokens = flatten_tokens[sorted_idxs]

        sorted_tokens = [self.model.tokenizer.decode(token_id) for token_id in  sorted_tokens[sorted_idxs]]
        fig = px.bar(x=sorted_tokens, y=sorted_activations)
        return fig
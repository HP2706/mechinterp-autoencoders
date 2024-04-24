import os
import tqdm
import pandas as pd
import torch
from jaxtyping import Float
from typing import List, Tuple, Optional, Any, Union, Literal
from torch.nn import functional as F
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset, Dataset
from autoencoder import AutoEncoder
from pydantic import BaseModel, field_validator
import plotly.express as px
from datamodels import MultiTokenActivationExample, ActivationExample
from automated_interpretability import AutomatedInterpretability
from utils import find_token_pos
import instructor
from openai import OpenAI

class PipelineConfig(BaseModel):
    device: str
    d_type: Any = torch.float32
    batch_size: int = 512
    seq_len: int = 128

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
        cfg : PipelineConfig,
        interpretability_model_name : str = 'gpt-4-turbo',
        save_path : str = "mechinterp_pipeline"
    ) -> None:

        self.automated_interp_pipeline = AutomatedInterpretability(
            instructor.from_openai(OpenAI()), model=interpretability_model_name
        )
        model = HookedTransformer.from_pretrained(model_name, device=cfg.device).to(cfg.d_type)
        
        self.dataset_name = dataset_name
        self.model = model
        self.encoder = AutoEncoder.load_from_hf(encoder_name, cfg.device).to(cfg.device)
    
        self.cfg = cfg
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def get_dataset(
        self,
        target_tokens: Optional[List[int]] = None, 
        shuffle: bool = True,
        n_sequences: Optional[int] = 100,
        sequence_len: int = 128,
        split: Literal["train", "validation"] = "train",
    ) -> Optional[torch.Tensor]:
        """
        Gets a batch of tokens where each sequence contains all tokens specified in target_tokens.
        We return None if no such sequence is found.
        """

        self.data : Dataset = load_dataset(self.dataset_name, split=split)# type: ignore

        if shuffle:
            self.data : Dataset = self.data.shuffle(seed=42) #type: ignore

        tokenized_data = utils.tokenize_and_concatenate(self.data, self.model.tokenizer, max_length=sequence_len) # type: ignore
        tokens : torch.Tensor = tokenized_data["tokens"] # type: ignore

        # Create a mask where each sequence that contains all target_tokens is True
        if target_tokens:
            contains_targets = torch.stack(
                [tokens == target_token for target_token in target_tokens]
            ).all(dim=0).any(dim=1)
            # Filter sequences that contain all target_tokens
            tokens = tokens[contains_targets]
        
        if n_sequences is not None:
            if len(tokens) > n_sequences:
                tokens = tokens[:n_sequences]
        

        if len(tokens) <= 1:
            return None
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

            hidden = self.encoder(mlp_acts, method='with_acts')

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
    ) -> torch.Tensor:
        
        _, cache = self.model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)] # shape (batch, seq_len, d_mlp)
        feature_acts = self.encoder(mlp_acts, method='with_acts')
        return feature_acts
    
    @torch.no_grad()
    def get_single_feature_acts(
        self,     
        tokens : torch.Tensor, 
        feature_index : int,
    ) -> torch.Tensor:
        '''gets the activation values for a specific feature index(index of the hidden layer)'''

        if tokens.device != self.cfg.device:
            tokens = tokens.to(self.cfg.device)

        _, cache = self.model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)] # shape (batch, seq_len, d_mlp)
        feature_in = self.encoder.W_enc[:, feature_index]
        feature_bias = self.encoder.b_enc[feature_index]
        feature_acts = F.relu((mlp_acts - self.encoder.b_dec) @ feature_in + feature_bias) # shape (batch, seq_len)
        feature_acts = feature_acts.cpu()
    
        return feature_acts
    
    def tune_batch_size(
        self,
        input : torch.Tensor, # shape (batch_size, seq_len)
    )-> int:
        
        assert len(input.shape) == 2, f"expected input to be 2D tensor of (batch_size, seq_len) got {input.shape}"
        nums_per_batch = self.cfg.batch_size * self.cfg.seq_len # the default seq_len
        inp_nums = input.numel()
        
        relative = nums_per_batch / inp_nums
        new_batch_size = int(input.shape[0] * relative)
        return new_batch_size

        
    @torch.no_grad()
    def create_acts_dataset(
        self, 
        feature_index: int,
        context_window: int = 4, # on both sides of token from anthropic paper
        dataset: Optional[torch.Tensor] = None,
        threshold: float = 0.01,
        entire_dataset: bool = False,
        with_entire_context: bool = False,
        sort: bool = False,
        remove_zeros: bool = True,
        save_dataset: bool = False
    ) -> Optional[pd.DataFrame]:
        '''gets the non zero activations for a feature index and the corresponding text
        Args:
            tokens: (batch_size, seq_len) or (seq_len)
            feature_index: int
        Returns:
            List[tuple[str, List[tuple[int, str, float]]]]: List of len batch_size of
            tuples where each tuple contains the text in the batch and the list of activation pairs for the feature index
            sorted by magnitude of activation
        '''
        
        # Check inputs
        if dataset is not None and entire_dataset:
            raise ValueError("Cannot specify both tokens and entire_dataset, to use entire dataset, set tokens to None")

        if dataset is None:
            if entire_dataset:
                dataset = self.get_dataset(
                    target_tokens = [feature_index], 
                    n_sequences = None, # get all sequences
                    sequence_len = (context_window*2)+1 
                ) 
            else:
                dataset = self.get_dataset(
                    target_tokens = [feature_index], 
                    n_sequences = self.cfg.batch_size
                )
        
        if dataset is None: # if no sequences contain the target token
            return None

        dataset_size = len(dataset)
        all_tokens = []
        all_activations = []
        tuned_batch_size = self.tune_batch_size(dataset)
        print("tuned batch size", tuned_batch_size)

        for batch_idx in tqdm.tqdm(range(0, dataset_size, tuned_batch_size)):
            batch_tokens = dataset[batch_idx:min(batch_idx + tuned_batch_size, dataset_size)]
            batch_activations = self.get_single_feature_acts(batch_tokens, feature_index)
            all_tokens.append(batch_tokens)
            all_activations.append(batch_activations)


        all_tokens = torch.cat(all_tokens, dim=0)
        all_activations = torch.cat(all_activations, dim=0)

        batch_interp = self.process_activations(
            all_tokens, all_activations, threshold, remove_zeros=remove_zeros
        )

        df = pd.DataFrame([example.model_dump() for example in batch_interp])
        if sort:
            df.sort_values("activation", ascending=False, inplace=True)

        if save_dataset:
            df.to_parquet(
                f"{self.save_path}/acts_feauture_{feature_index}_from_dataset_size_{dataset_size}.parquet"
            )
        
        return df

    def process_activations(
        self, 
        dataset, 
        activations, 
        threshold, 
        remove_zeros: bool
    ) -> Union[List[ActivationExample], List[MultiTokenActivationExample]]:
        batch_results = []

        for batch_idx in tqdm.trange(len(activations)):
            batch_tokens = dataset[batch_idx]
            batch_activations = activations[batch_idx]
            if remove_zeros:
                acts, zero_tokens = self.filter_non_zero_sequence(
                    batch_tokens, batch_activations, threshold
                ) 
                if acts.numel() == 0:  # no non-zero activations, we skip
                    continue   
            
                for (token_id, act) in zip(zero_tokens.tolist(), acts.tolist()):
                    idxs = find_token_pos(token_id, batch_tokens)
                    # find the position in the original sequence where the token appears
                    
                    print("idxs", idxs)
                    context = ''
                    start_idx = 0
                    token_str = self.model.tokenizer.decode([token_id])
                    for idx in idxs:
                        chunk = ''.join(self.model.tokenizer.batch_decode(batch_tokens[start_idx:idx]))
                        print("chunk", chunk)
                        context += chunk + f'|{token_str}|'
                        start_idx = idx + 1

                    chunk = ''.join(self.model.tokenizer.batch_decode(batch_tokens[start_idx:])) # get the last chunk
                    context += chunk

                    print("context", context)

                    batch_results.append(
                        ActivationExample(
                            token=token_str, 
                            token_id=token_id, 
                            positions = idxs,
                            activation=act,
                            context=context,
                        )
                    )
            else:
                text_tokens = self.model.tokenizer.batch_decode(batch_tokens)
                batch_results.append(
                    MultiTokenActivationExample(
                        tokens=text_tokens, 
                        token_ids=batch_tokens.tolist(), 
                        activation=batch_activations.tolist(),
                        context=''.join(self.tokens_to_str(batch_tokens))
                    )
                )
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

    


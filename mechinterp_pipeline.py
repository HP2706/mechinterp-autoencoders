from functools import partial
from copy import deepcopy
import os
import time
import tqdm
import pandas as pd
import numpy as np
import torch
from jaxtyping import Float
from typing import Callable, List, Tuple, Optional, Any, Union, Literal, Dict
from torch.nn import functional as F
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset, Dataset
from autoencoder import AutoEncoder
from pydantic import BaseModel, field_validator
import plotly.express as px
from datamodels import MultiTokenActivationExample, ActivationExample, InterpretabilityData
from automated_interpretability import AutomatedInterpretability
from utils import find_token_pos, filter_zeros, filter_non_zero_sequence
from mechninterp_utils import torch_spearman_correlation
import instructor
import multiprocessing as mp
from openai import OpenAI
torch.manual_seed(42)
np.random.seed(42)

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
        self.tokenizer = deepcopy(model.tokenizer)
        self.encoder = AutoEncoder.load_from_hf(encoder_name, cfg.device).to(cfg.device)
        self.all_tokens = None
        self.cfg = cfg
        self.save_path = save_path
        self.interp_save_path = os.path.join(save_path, "interpretability_data.parquet")
        if os.path.exists(self.interp_save_path):
            self.interp_df = pd.read_parquet(self.interp_save_path)
        else:
            self.interp_df = None


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

        if self.all_tokens is None:
            self.data : Dataset = load_dataset(self.dataset_name, split=split)# type: ignore
            tokenized_data = utils.tokenize_and_concatenate(self.data, self.tokenizer, max_length=sequence_len) # type: ignore
            tokens : torch.Tensor = tokenized_data["tokens"] # type: ignore
            self.all_tokens = tokens

        if shuffle:
            tokens = self.all_tokens[torch.randperm(len(self.all_tokens))]
        
        tokens = self.all_tokens.clone()
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

            _ , cache = self.model.run_with_cache(
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
    def get_single_neuron_acts(
        self,
        tokens : torch.Tensor,
        neuron_index : int,
    ) -> torch.Tensor:
        '''gets the activation values for a specific neuron index(index of the hidden layer)'''

        if tokens.device != self.cfg.device:
            tokens = tokens.to(self.cfg.device)

        _, cache = self.model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)]
        neuron_acts = mlp_acts[:, :, neuron_index] # shape (batch, seq_len)
        return neuron_acts.cpu()


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
        neuron_or_feature : Literal["neuron", "feature"],
        index: int,
        target_tokens : Optional[List[int]] = None,
        context_window: int = 4, # on both sides of token from anthropic paper
        dataset: Optional[torch.Tensor] = None,
        threshold: float = 0.01,
        entire_dataset: bool = False,
        sort: bool = True,
        remove_zeros: bool = True,
        save_dataset: bool = True,
        n_sequences: Optional[int] = None,
        seq_len: int = 9
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
                    target_tokens = target_tokens, 
                    n_sequences = None, # get all sequences
                    sequence_len = (context_window*2)+1 
                ) 
            else:
                dataset = self.get_dataset(
                    target_tokens = target_tokens, 
                    n_sequences = n_sequences,
                    sequence_len = seq_len
                )
        
        if dataset is None: # if no sequences contain the target token
            return None

        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size} features: {index}")
        all_tokens = []
        all_activations = []
        tuned_batch_size = self.tune_batch_size(dataset)
        for batch_idx in tqdm.tqdm(range(0, dataset_size, tuned_batch_size), desc="getting activations"):
            batch_tokens = dataset[batch_idx:min(batch_idx + tuned_batch_size, dataset_size)]
            if neuron_or_feature == "neuron":
                batch_activations = self.get_single_neuron_acts(batch_tokens, index)
            else:   
                batch_activations = self.get_single_feature_acts(batch_tokens, index)
            all_tokens.append(batch_tokens)
            all_activations.append(batch_activations)

        all_tokens = torch.cat(all_tokens, dim=0)
        all_activations = torch.cat(all_activations, dim=0)

        batch_interp = self.parallel_process_activations(
            neuron_or_feature, index, all_tokens, all_activations, threshold, remove_zeros=remove_zeros
        )

        df = pd.DataFrame([example.model_dump() for example in batch_interp])
        if sort:
            t0 = time.time()
            df.sort_values("activation", ascending=False, inplace=True)
            print(f"Sorting took {time.time() - t0:.2f} seconds")
        if save_dataset:
            df.to_parquet(
                f"{self.save_path}/acts_{neuron_or_feature}_{index}_from_dataset_size_{dataset_size}.parquet"
            )
        
        return df



    def process_activations( 
        self, 
        neuron_or_feature : Literal["neuron", "feature"],
        index : int,
        dataset, 
        activations, 
        threshold, 
        remove_zeros: bool
    ) -> Union[List[ActivationExample], List[MultiTokenActivationExample]]:
        batch_results = []

        #we build a massive in memory lookup table
        decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in dataset.view(-1).unique().tolist()]
        token_to_str_map = dict(zip(dataset.view(-1).unique().tolist(), decoded_tokens))

        if remove_zeros:
            activations = filter_zeros(activations, threshold) 

        for batch_idx in tqdm.trange(len(activations), desc="Processing activations"):
            batch_tokens = dataset[batch_idx]
            batch_activations = activations[batch_idx]
            if remove_zeros:
                acts, zero_tokens = filter_non_zero_sequence(
                    batch_tokens, batch_activations, threshold
                ) 
                if acts.numel() == 0:  # no non-zero activations, we skip
                    continue   
            
                for (token_id, act) in zip(zero_tokens.tolist(), acts.tolist()):
                    idxs = find_token_pos(token_id, batch_tokens)
                    # find the position in the original sequence where the token appears
                    
                    context = ''
                    start_idx = 0
                    token_str = token_to_str_map[token_id]
                    for idx in idxs:
                        chunk = ''.join(self.tokenizer.batch_decode(batch_tokens[start_idx:idx])) # type: ignore
                        context += chunk + f'|{token_str}|'
                        start_idx = idx + 1

                    chunk = ''.join(self.tokenizer.batch_decode(batch_tokens[start_idx:])) #type: ignore get the last chunk
                    context += chunk

                    batch_results.append(
                        ActivationExample(
                            neuron_or_feature=neuron_or_feature,
                            index=index,
                            token=token_str, 
                            token_id=token_id, 
                            positions = idxs,
                            activation=act,
                            context=context,
                        )
                    )
            else:
                text_tokens = self.tokenizer.batch_decode(batch_tokens)
                batch_results.append(
                    MultiTokenActivationExample(
                        neuron_or_feature=neuron_or_feature,
                        index=index,
                        tokens=text_tokens, 
                        token_ids=batch_tokens.tolist(), 
                        activation=batch_activations.tolist(),
                        context=''.join(self.tokens_to_str(batch_tokens))
                    )
                )
        return batch_results

    def build_and_interpret(self, indices : List[int], kwargs):
        for idx in tqdm.tqdm(indices, desc="Creating Datasets"):
            self.create_acts_dataset(index =idx, **kwargs, save_dataset=True)
            
    
    def get_interpretability_correlation(
        self, 
        feature_or_neuron : Literal['feature', 'neuron'],
        index : int,
        total_examples : int = 60
    ):
        path_start = f'acts_{feature_or_neuron}_{index}_from'
        paths = [
            path for path in os.listdir('mechinterp_pipeline')
            if path.endswith('.parquet') and path.startswith(path_start)
        ] 
        
        if len(paths) == 0:
            raise ValueError(f"No dataset found for {feature_or_neuron} {index}")


        df = pd.read_parquet(os.path.join('mechinterp_pipeline', paths[0]))

        #filter duplicates

        if len(df) < 100:
            raise ValueError(f"Dataset for {feature_or_neuron} {index} is too small, need at least 100 examples")
    
        df['interval'] = pd.qcut(df['activation'], q=12, labels=False)
        
        dataset = {}

        #we get 2 random examples for each interval and 10 for the last one(the biggest)
        selected_indices = []

        for i in range(12):
            quantile_indices = df[df['interval'] == i].index
            sample_size = min(10 if i == 11 else 2, len(quantile_indices))
            sampled_indices = df.loc[quantile_indices].sample(n=sample_size, random_state=42).index
            dataset[f'quantile_{i}'] = [
                ActivationExample(**row) for row in df.loc[sampled_indices].to_dict(orient='records') #type: ignore
            ]
            selected_indices.extend(sampled_indices.tolist())

        # Now, select random samples from the indices that were not picked
        remaining_indices = list(set(df.index) - set(selected_indices))
        random_samples = df.loc[np.random.choice(remaining_indices, size=5, replace=False)]
        dataset['random'] = [
            ActivationExample(**row) for row in random_samples.to_dict(orient='records') #type: ignore
        ]
        feature_hypothesis = self.automated_interp_pipeline.explain_activation(dataset)
        left_out = df.drop(selected_indices)


        #get random 30 examples or the max number under 30
        left_out = left_out.sample(n=min(total_examples, len(left_out)), random_state=42)


        predictions = []
        labels = []
        examples_per_pred = 8

        for i in tqdm.tqdm(range(0, len(left_out), examples_per_pred), desc="llm predicting activations"):
            batch = left_out.iloc[i:i+examples_per_pred]
            lst = [ActivationExample(**row.to_dict()) for _ , row in batch.iterrows()]
            llm_predictions = self.automated_interp_pipeline.predict_activation(
                unseen_examples=lst,
                hypothesis=feature_hypothesis,
                feature_or_neuron=feature_or_neuron,    
            )
            predictions.extend([prediction.value for prediction in llm_predictions])
            labels.extend([actual.activation for actual in lst])

        spearman_corr = torch_spearman_correlation(
            torch.tensor(predictions, dtype=torch.float), torch.tensor(labels, dtype=torch.float)
        ).item()
    
        data = InterpretabilityData(
            feature_or_neuron=feature_or_neuron, 
            index=index, 
            llm_explanation=feature_hypothesis.hypothesis, 
            llm_predictions=predictions,
            actual_data=labels,
            spearman_corr=spearman_corr
        )
        self.interp_df = pd.concat([self.interp_df, pd.DataFrame([data.model_dump()])]) #type: ignore
        self.interp_df.to_parquet(self.interp_save_path)

    def parallel_process_activations(
        self, 
        neuron_or_feature: Literal["neuron", "feature"],
        index: int,
        dataset: torch.Tensor, 
        activations: torch.Tensor, 
        threshold: float, 
        remove_zeros: bool
    ) -> List[Any]:
        
        n_processes = min(os.cpu_count(), 8)  #type: ignore
        batch_size = len(dataset) // n_processes  # Define batch size based on number of processes
        print(f"Batch size: {batch_size}")

        tokenizer_copy = deepcopy(self.tokenizer) 
        with mp.Pool(processes=n_processes) as pool:
            results = pool.starmap(
                process_batch,
                tqdm.tqdm(
                    [
                        (
                            neuron_or_feature, 
                            index, 
                            dataset[i:i + batch_size], 
                            activations[i:i + batch_size], 
                            threshold, 
                            remove_zeros,
                            filter_non_zero_sequence,
                            tokenizer_copy,
                        )
                        for i in range(0, len(dataset), batch_size)
                    ],
                    desc="Processing batches"
                )
            )

        batch_results = []
        for sublist in results:
            batch_results.extend(sublist)

        return batch_results


def process_batch(
    neuron_or_feature: Literal["neuron", "feature"],
    index: int, 
    dataset: torch.Tensor, 
    activations: torch.Tensor, 
    threshold: float, 
    remove_zeros: bool,
    filter_non_zero_sequence : Callable[
        [torch.Tensor, torch.Tensor, Optional[float]], tuple[torch.Tensor, torch.Tensor]
    ], 
    tokenizer : Any,
) -> List[Any]:
    assert len(dataset) == len(activations), f"expected dataset and activations to have the same length, got {len(dataset)} and {len(activations)}"
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in dataset.view(-1).unique().tolist()] #type: ignore
    token_to_str_map = dict(zip(dataset.view(-1).unique().tolist(), decoded_tokens))

    if remove_zeros:
        activations = filter_zeros(activations, threshold) 

    batch_results = []
    for batch_idx in tqdm.trange(len(activations), desc="Processing activations"):
        batch_tokens = dataset[batch_idx]
        batch_activations = activations[batch_idx]
        if remove_zeros:
            acts, zero_tokens = filter_non_zero_sequence(
                batch_tokens, batch_activations, threshold
            ) 
            if acts.numel() == 0:  # no non-zero activations, we skip
                continue   
        
            for (token_id, act) in zip(zero_tokens.tolist(), acts.tolist()):
                idxs = find_token_pos(token_id, batch_tokens)
                # find the position in the original sequence where the token appears
                
                context = ''
                start_idx = 0
                token_str = token_to_str_map[token_id]
                for idx in idxs:
                    chunk = ''.join(tokenizer.batch_decode(batch_tokens[start_idx:idx])) # type: ignore
                    context += chunk + f'|{token_str}|'
                    start_idx = idx + 1

                chunk = ''.join(tokenizer.batch_decode(batch_tokens[start_idx:])) #type: ignore get the last chunk
                context += chunk

                batch_results.append(
                    ActivationExample(
                        neuron_or_feature=neuron_or_feature,
                        index=index,
                        token=token_str, 
                        token_id=token_id, 
                        positions = idxs,
                        activation=act,
                        context=context,
                    )
                )
        else:
            text_tokens = tokenizer.batch_decode(batch_tokens)
            batch_results.append(
                MultiTokenActivationExample(
                    neuron_or_feature=neuron_or_feature,
                    index=index,
                    tokens=text_tokens, 
                    token_ids=batch_tokens.tolist(), 
                    activation=batch_activations.tolist(),
                    context=''.join(tokenizer.batch_decode(batch_tokens))
                )
            )
    return batch_results
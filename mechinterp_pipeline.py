from copy import deepcopy
import os
import time
import tqdm
import pandas as pd
import numpy as np
import torch
from typing import Callable, List, Tuple, Optional, Any, Union, Literal, Dict
from torch.nn import functional as F
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset, Dataset
from autoencoder import AutoEncoder
from pydantic import BaseModel
from common import image, stub, vol, PATH
from datamodels import (
    MultiTokenActivationExample, 
    ActivationExample, 
    InterpretabilityData, 
    FeatureDescription, 
    FeatureSample,
    PipelineConfig
)
from automated_interpretability import AutomatedInterpretability
from utils import (
    find_token_pos, 
    filter_zeros, 
    filter_non_zero_sequence, 
    convert_to_pydantic_model,
    remove_keys, 
    write_models_to_json, 
    load_models_from_json,
    time_decorator
)
from mechninterp_utils import torch_spearman_correlation
import instructor
import multiprocessing as mp
from modal import gpu, Secret, enter, method
from openai import OpenAI
torch.manual_seed(42)
np.random.seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@stub.cls(
    image=image,
    volumes={PATH: vol},
    timeout=60*60, #1 hour
    cpu=20, #20 cores    
    gpu=gpu.T4(),
    secrets=[Secret.from_name("my-openai-secret")],
    concurrency_limit=20
)
class MechInterpPipeline:
    def __init__(
        self,
        model_name : str,
        encoder_name : str,
        dataset_name : str,
        cfg : PipelineConfig,
        interpretability_model_name : str = 'gpt-4-turbo',
        folder_name : str = "mechinterp_pipeline"
    ) -> None:
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.interpretability_model_name = interpretability_model_name
        self.save_path = f"{PATH}/{folder_name}"
    
    @enter()
    def load(self):
        if torch.cuda.is_available():
            self.cfg.device = "cuda" #override the device to cuda if available

        self.automated_interp_pipeline = AutomatedInterpretability(
            instructor.from_openai(OpenAI()), model=self.interpretability_model_name
        )
        model : HookedTransformer = HookedTransformer.from_pretrained(
            self.model_name, device=self.cfg.device
        ).to(self.cfg.d_type) #type: ignore
        
        self.model = model 
        self.tokenizer = deepcopy(model.tokenizer)
        self.encoder = AutoEncoder.load_from_hf(self.encoder_name, self.cfg.device).to(self.cfg.device)
        self.all_tokens = None
        
        #TODO is this a smart design decision?
        self.interp_save_path = os.path.join(self.save_path, "interpretability_data.parquet")
        if os.path.exists(self.interp_save_path):
            self.interp_df = pd.read_parquet(self.interp_save_path)
        else:
            self.interp_df = None

        self.feature_df_save_path = os.path.join(self.save_path, "features.json")
        if os.path.exists(self.feature_df_save_path):
            self.feature_data : List[FeatureDescription] = load_models_from_json(FeatureDescription, self.feature_df_save_path)
        else:
            self.feature_data = [] 

        os.makedirs(self.save_path, exist_ok=True)

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
    @time_decorator
    def create_acts_dataset(
        self, 
        feature_or_neuron : Literal["neuron", "feature"],
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
        t0 = time.time()
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
            if feature_or_neuron == "neuron":
                batch_activations = self.get_single_neuron_acts(batch_tokens, index)
            else:   
                batch_activations = self.get_single_feature_acts(batch_tokens, index)
            all_tokens.append(batch_tokens)
            all_activations.append(batch_activations)

        all_tokens = torch.cat(all_tokens, dim=0)
        all_activations = torch.cat(all_activations, dim=0)

        batch_interp = self.parallel_process_activations(
            feature_or_neuron, index, all_tokens, all_activations, threshold, remove_zeros=remove_zeros
        )

        df = pd.DataFrame([example.model_dump() for example in batch_interp])
        if sort:
            t0 = time.time()
            df.sort_values("activation", ascending=False, inplace=True)
            print(f"Sorting took {time.time() - t0:.2f} seconds")
        if save_dataset:
            df.to_parquet(
                f"{self.save_path}/acts_{feature_or_neuron}_{index}_from_dataset_size_{dataset_size}.parquet"
            )
            vol.commit()
            print(f"Saved dataset to {self.save_path}")
            print("checking if dataset is saved", os.listdir(self.save_path))
        return df

    @method()
    def build_and_interpret(self, idx:int, kwargs):
            self.create_acts_dataset(index =idx, **kwargs, save_dataset=True)
            neuron_feature = kwargs.get('feature_or_neuron')
            #we interpret the data
            self.get_interpretability_correlation(feature_or_neuron= neuron_feature, index=idx)   
    
    def get_interpretability_correlation(
        self, 
        feature_or_neuron : Literal['feature', 'neuron'],
        index : int,
    ):
        path_start = f'acts_{feature_or_neuron}_{index}_from'
        paths = [
            path for path in os.listdir(self.save_path)
            if path.endswith('.parquet') and path.startswith(path_start)
        ]

        print(f"Found {len(paths)} datasets for {feature_or_neuron} {index}")
        
        if len(paths) == 0:
            raise ValueError(f"No dataset found for {feature_or_neuron} {index}")

        df = pd.read_parquet(os.path.join(self.save_path, paths[0]))

        if len(df) < 100:
            raise ValueError(f"Dataset for {feature_or_neuron} {index} is too small, need at least 100 examples")

        #quantize 
        # Normalize the 'activation' column to range between 0 and 9
        min_activation = df['activation'].min()
        max_activation = df['activation'].max()
        
        # Define the number of bins
        num_bins = 9
        # Create bins from min to max activation
        bins = np.linspace(min_activation, max_activation, num_bins + 1)
        bin_labels = np.arange(num_bins)  # Labels for each bin


        if 'context' in df.columns: #THIS IS A HACK
            df.rename(columns={'context': 'text'}, inplace=True)

        # Use pd.cut to bin the data
        df['quantized_activation'] = pd.cut(df['activation'], bins=bins, labels=bin_labels, include_lowest=True) # type: ignore
        dataset = {}

        #we get 2 random examples for each interval and 10 for the last one(the biggest)
        selected_indices = []

        for i in range(12):
            quantile_indices = df[df['quantized_activation'] == i].index
            sample_size = min(10 if i == 11 else 2, len(quantile_indices))
            sampled_indices = df.loc[quantile_indices].sample(n=sample_size, random_state=42).index
            dataset[f'quantile_{i}'] = [
                {key: value for key, value in row.items() if key != 'activation'} 
                for row in df.loc[sampled_indices].to_dict(orient='records')
            ]
            selected_indices.extend(sampled_indices.tolist())

        # Now, select random samples from the indices that were not picked
        remaining_indices = list(set(df.index) - set(selected_indices))
        random_samples = df.loc[np.random.choice(remaining_indices, size=5, replace=False)]
        dataset['random'] = [
            remove_keys(row, 'activation')
            for row in random_samples.to_dict(orient='records')
        ]
        feature_hypothesis = self.automated_interp_pipeline.explain_activation(dataset)

        feature_description = FeatureDescription(
            **feature_hypothesis.model_dump(),
            feature_or_neuron=feature_or_neuron,
            index=index,
            high_act_samples=[
                convert_to_pydantic_model(FeatureSample, row) 
                for row in df.sort_values("activation", ascending=False).head(5).to_dict(orient='records')
            ],
            low_act_samples=[
                convert_to_pydantic_model(FeatureSample, row) 
                for row in df.sort_values("activation", ascending=False).tail(5).to_dict(orient='records')
            ],
        )
        self.feature_data.append(feature_description)
        df = df.drop(selected_indices) # remove the selected indices from the dataframe

        #we get:
        #top 6 activating examples
        #2 for each 12 quantiles
        #10 completely random
        #IGNORE 20 top activating tokens out of context TODO what is meant here??

        top_6 = df.sort_values("activation", ascending=False).head(6)
        top_6_indices = top_6.index.tolist()

        # Select 2 examples from each of the 12 quantiles
        quantile_indices = []
        num_quantiles = 12
        quantile_size = 2
        for q in range(num_quantiles):
            quantile = df[df['quantized_activation'] == q]
            if len(quantile) >= quantile_size:
                sampled = quantile.sample(n=quantile_size, random_state=42)
                quantile_indices.extend(sampled.index.tolist())

        # Select 10 completely random examples, excluding already selected indices

        available_df = df.drop(index=top_6_indices + quantile_indices)
        random_samples = available_df.sample(n=10, random_state=42)

        examples_df = pd.concat([top_6, df.loc[quantile_indices], random_samples])

        lst = [
            remove_keys(row, ['activation', 'quantized_activation']) 
            # removes the keys from the row(these are hidden from the model so it doesnt use them in its prediction)
            for row in examples_df.to_dict(orient='records')
        ]
        # pre
        batch_size = 8
        
        labels : List[int] = []
        predictions : List[int] = []
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i+batch_size] 
          
            llm_predictions = self.automated_interp_pipeline.predict_activation(
                unseen_examples=batch,
                hypothesis=feature_hypothesis,
                feature_or_neuron=feature_or_neuron, 
                max_tries=2   
            )
            if llm_predictions is None:
                continue

            labels.extend([row['quantized_activation'] for row in examples_df[i:i+batch_size].to_dict(orient='records')])
            predictions.extend([pred.value for pred in llm_predictions])

        assert len(labels) == len(predictions), f"expected labels and predictions to have the same length, got {len(labels)} and {len(predictions)}"
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
        write_models_to_json(self.feature_data, self.feature_df_save_path)
        vol.commit()

    def parallel_process_activations(
        self, 
        feature_or_neuron: Literal["neuron", "feature"],
        index: int,
        dataset: torch.Tensor, 
        activations: torch.Tensor, 
        threshold: float, 
        remove_zeros: bool
    ) -> List[Any]:
        
        n_processes = min(os.cpu_count(), 6)  #type: ignore
        batch_size = len(dataset) // n_processes  # Define batch size based on number of processes
        print(f"Batch size: {batch_size}")
        print(f"Number of processes: {n_processes}")

        tokenizer_copy = deepcopy(self.tokenizer) 
        with mp.Pool(processes=n_processes) as pool:
            results = pool.starmap(
                process_batch,
                tqdm.tqdm(
                    [
                        (
                            feature_or_neuron, 
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
    feature_or_neuron: Literal["neuron", "feature"],
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
                        feature_or_neuron=feature_or_neuron,
                        index=index,
                        token=token_str, 
                        token_id=token_id, 
                        positions = idxs,
                        activation=act,
                        text=context,
                    )
                )
        else:
            text_tokens = tokenizer.batch_decode(batch_tokens)
            batch_results.append(
                MultiTokenActivationExample(
                    feature_or_neuron=feature_or_neuron,
                    index=index,
                    tokens=text_tokens, 
                    token_ids=batch_tokens.tolist(), 
                    activation=batch_activations.tolist(),
                    text=''.join(tokenizer.batch_decode(batch_tokens))
                )
            )
    return batch_results
import os
import random
import tqdm
import shutil
import pandas as pd
import numpy as np
import torch
from utils import get_device, load_feature_descriptions, write_to_json, flatten_lst
from typing import Any, Dict, List, Literal, Optional, Type, Union
from utils import filter_valid_image_urls
import time
from autoencoder import (
    AutoEncoder, 
    GatedAutoEncoder,
    AutoEncoderBase,
    AutoencoderConfig,
)
from common import (
    image, 
    stub, 
    vol, 
    PATH, 
    LAION_DATASET_PATH, 
    dataset_vol, 
    EMB_FOLDER, 
    METADATA_FOLDER
)
from datamodels import (
    ImageContent,
    FeatureDescription, 
    FeatureSample,
    InterpretabilityMetaData,
    PipelineConfig,
    save_html
)
from automated_interpretability import AutomatedInterpretability
from utils import (
    write_to_json, 
    load_feature_descriptions,
    filter_non_zero_batch,
)
from litellm import completion
from Laion_Processing.dataloader import LaionDataset
import instructor
from modal import gpu, Secret, enter, method, build
from openai import OpenAI

torch.manual_seed(42)
np.random.seed(42)
#set seed for pandas
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@stub.cls(
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},
    image = image,
    gpu=gpu.A10G(),
    timeout=10*60*60, #10 hours   
    secrets=[Secret.from_name("my-gemini-secret"), Secret.from_name("my-openai-secret")], 
)
class ClipMechInterpPipeline:
    def __init__(
        self,
        auto_encoder_path_dir: str,    
        device : Optional[Literal['cpu', 'cuda']] = None,
        interpretability_model_name : str = 'gpt-4-turbo',
        folder_name : str = "clip_mechinterp_pipeline",
        **dataset_kwargs,
    ):
        self.model = AutoEncoderBase.load_from_checkpoint(auto_encoder_path_dir)
        self.interpretability_model_name = interpretability_model_name
        self.save_path = f"{PATH}/{folder_name}"
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        os.makedirs(self.save_path, exist_ok=True)

        self.dataset = LaionDataset(**dataset_kwargs)
        self.automated_interp_pipeline = AutomatedInterpretability(
            OpenAI(), model=self.interpretability_model_name
            #instructor.from_litellm(completion, instructor.mode.Mode.MD_JSON), model=self.interpretability_model_name
        )
        #TODO is this a smart design decision?
        self.interp_vis_save_path = os.path.join(self.save_path, f"html_vis_{self.model.dir_name}")
        os.makedirs(self.interp_vis_save_path, exist_ok=True)
        self.interp_save_path = os.path.join(self.save_path, f"interpretability_data_{self.model.dir_name}.parquet")
        if os.path.exists(self.interp_save_path):
            self.interp_df = pd.read_parquet(self.interp_save_path)
        else:
            self.interp_df = None

        self.feature_df_save_path = os.path.join(self.save_path, "features.json")
        if os.path.exists(self.feature_df_save_path):
            self.feature_data : Dict[int,FeatureDescription] = load_feature_descriptions(FeatureDescription, self.feature_df_save_path)
        else:
            self.feature_data = {} 



    def get_acts_dir(self, index : Union[int, Literal['all']]) -> str:
        if index == 'all':
            return f"{PATH}/laion_acts_all_{self.model.dir_name}"
        else:
            return f"{PATH}/laion_acts_idx_{index}_{self.model.dir_name}"

    def get_activations_metadata(
        self, 
        index : Union[int, Literal['all']] = 'all'
    ) -> pd.DataFrame:
        
        dirname = self.get_acts_dir(index)
        if not os.path.exists(dirname):
            dirname = f"{PATH}/laion_acts_all_{self.model.dir_name}"
            
        if not os.path.exists(dirname):
            raise ValueError(f"Index {index} does not exist")
        
        dataframes = [
            pd.read_parquet(os.path.join(dirname, file)) 
            for file in os.listdir(dirname) 
            if file.endswith(".parquet")
        ]
        df = pd.concat(dataframes)
        if isinstance(index, int):
            df = df[df['feature_idx'] == index]
        return df

    @method()
    def delete_dir(
        self,
        index : int
    ):
        dirname = self.get_acts_dir(index)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
            vol.commit()

    @method()
    #NOTE for debugging
    def check_activations(self):

        for (tensor, df_metadata) in tqdm.tqdm(
            self.dataset.iter_files(max_count=1), 
        ):      
            batch_size = 512
            step = 0

            for j in range(0, tensor.shape[0], batch_size):
                batch = tensor[j:j+batch_size].to(self.device)
                data = self.model.forward(batch, 'with_loss')
                activations = data.acts
                recons_loss = (data.x_reconstruct - (batch- batch.mean(dim=0))).pow(2).mean() 
                print("\nrecons_loss", recons_loss)
                print("mean norm", batch.norm(dim=1).mean())
                print("\nmax", activations.max(), "min", activations.min(), "mean", activations.mean())
                # Get the max activation for each index across the batch
                max_activations = activations.max(dim=0)[0]
                # Compute the mean of these max activations
                mean_max_activations = max_activations.mean()
                print("\nmean_max_activations", mean_max_activations)
                print("\nmean_max_activations.item()", mean_max_activations.item())
                step += 1

                if step > 5:
                    break

    @method()
    def create_acts_dataset(
        self,
        n_files : int = 5,
    ) -> None:

        dirname = f"{PATH}/laion_acts_all_{self.model.dir_name}"
        os.makedirs(dirname, exist_ok=True)

        dataframes : List[pd.DataFrame] = []

        def save_intermediate():
            #save intermediate 
            df = pd.concat(dataframes)
            df.to_parquet(f"{dirname}/metadata.parquet")
            vol.commit()


        nrows = 0
        for (tensor, df_metadata) in tqdm.tqdm(
            self.dataset.iter_files(max_count=n_files), 
            total=n_files,
            desc="Processing Files"
        ):            
            
            with torch.no_grad():
                df_rows = []
                batch_size = 1024
                for j in tqdm.tqdm(range(0, len(tensor), batch_size)):
                    scaled_batch = tensor[j:j+batch_size].to(self.device)
                    out = self.model.forward(scaled_batch, 'with_loss')
                    non_zero_indices, _ = filter_non_zero_batch(out.acts, threshold=1e-3)
                    
                    #if there are no non-zero activations, we skip the batch because 
                    # it is all zeros or below the activation threshold
                    if non_zero_indices.nelement() == 0:
                        continue
                    
                    # Get non-zero activations and their indices for the entire batch
                    non_zero_activations = out.acts[non_zero_indices]
                    non_zero_positions = (non_zero_activations != 0).nonzero(as_tuple=False)
                    original_indices = (non_zero_indices + j).tolist()
                    
                    # Extract the activation values using these indices
                    activation_values = non_zero_activations[non_zero_positions[:, 0], non_zero_positions[:, 1]]
            
                    #this might become the bottleneck
                    for idx, value in zip(non_zero_positions.tolist(), activation_values.tolist()):
                        df_rows.append(
                            {**df_metadata.iloc[original_indices[idx[0]]].to_dict(), 
                            'activation': value,
                            'feature_idx': idx[1],
                            'data_idx': original_indices[idx[0]]+nrows,
                        })
                        

            dataframes.append(pd.DataFrame(df_rows))
            print("df_rows", len(df_rows))
            print("df head", dataframes[-1].head())
            nrows += len(df_metadata)

        df = pd.concat(dataframes)
        num_bins = 9
        activations = df['activation']
        df['quantized_acts'] = np.digitize(activations, bins = np.linspace(activations.min(), activations.max(), num_bins))
        save_intermediate()
        vol.commit()

    @method()
    #NOTE this will almost never be used because activations are so sparse that it can be done all at once.
    def create_acts_dataset_by_index(
        self,
        index : int,
        n_files : int = 5,
    ) -> None:
        
        dirname = self.get_acts_dir(index)
        os.makedirs(dirname, exist_ok=True)

        dataframes = []
        activations = []

        for (tensor, df_metadata) in tqdm.tqdm(
            self.dataset.iter_files(max_count=n_files), 
            total=n_files,
            desc="Processing Files"
        ):            
            with torch.no_grad():
                batch = []
                removed_indices = []
                batch_size = 512*20*500 # note we are only using one weight idx, so 1000x less compute and memory
                for j in range(0, tensor.shape[0], batch_size):
                    batch_tensor = tensor[j:j+batch_size].to(self.device)
                    result = self.model.get_single_feature_acts(batch_tensor, feature_index=index).cpu()
                    non_zero_indices, zero_indices = filter_non_zero_batch(result) # be very careful with thresholding
                    removed_indices.extend(
                        (zero_indices + j).tolist() # get the global index
                    )
                    batch.append(result[non_zero_indices])

                dataframes.append(df_metadata.drop(removed_indices))
                filtered_acts = torch.cat(batch)
                if len(filtered_acts) == 0: 
                    return  # or handle the case as needed
                activations.append(filtered_acts)

        activations = torch.cat(activations)
        if len(activations) < 50:
            print("Not enough activations to process, got", len(activations))
            return
        
        df = pd.concat(dataframes)
        df['activation'] = activations
        num_bins = 9
        df['quantized_acts'] = np.digitize(activations, bins = np.linspace(activations.min(), activations.max(), num_bins))
        
        df.to_parquet(f"{dirname}/metadata.parquet")
        vol.commit()

    @method()
    def get_interpretability_explanation(
        self, 
        index: int,
        feature_or_neuron: Literal['feature', 'neuron'] = 'feature',
    ):
        df = self.get_activations_metadata(index).sort_values(by='activation', ascending=False)
        # Process each quantization bin
        data_by_quant, remaining_indices = sample_and_filter_data(df)

        # Filter positive and negative samples based on activation values
        positive_samples = [
            lst for (key, lst) in data_by_quant.items() if isinstance(key, int) and key > 5    
        ]
        negative_samples = [
            lst for (key, lst) in data_by_quant.items() if isinstance(key, int) and key < 5    
        ]

        # Flatten the list of FeatureSample objects for processing
        all_samples = flatten_lst([lst for lst in data_by_quant.values()])
        # Process and format data
        print("number of samples", len(all_samples))
        random.shuffle(all_samples) #inplace
        valid_samples = filter_valid_image_urls([elm.content.image_url for elm in all_samples]) #type: ignore
        if sum(valid_samples) != len(all_samples): # check if all are True
            raise ValueError("Not all samples are valid, filtering doesnt work", valid_samples)
        
        save_html(all_samples[:5], os.path.join(self.interp_vis_save_path, f"test_feature_idx_{index}.html"))
        vol.commit()
        feature_hypothesis = self.automated_interp_pipeline.explain_activation(all_samples[:3])

        print("feature_hypothesis", feature_hypothesis)
        # Create and save feature description
        feature_description = FeatureDescription.build_feature_description(
            feature_hypothesis, 
            index, 
            feature_or_neuron, 
            flatten_lst(positive_samples), 
            flatten_lst(negative_samples)
        )
        
        self.feature_data[index] = feature_description #NOTE this will overwrite the previous 
        #feature description, perhaps TODO make warning if this happens
        write_to_json(self.feature_data, self.feature_df_save_path)
        vol.commit()

    @method()
    def get_interpretability_correlation(
        self,
        feature_index: int,
        n_samples: int = 100,
    ): 
        df = self.get_activations_metadata(feature_index)
        df.sort_values(by='activation', ascending=False, inplace=True)
        print("columns", df.columns)
        print(df.head())
        if feature_index not in self.feature_data:
            raise ValueError(f"Feature {feature_index} does not exist in feature_data")
        
        sampled_data_by_quant, remaining_indices = sample_and_filter_data(df)
        # we take 10 randomly for each quantile
        data = flatten_lst([lst for lst in sampled_data_by_quant.values()])
        hypothesis = self.feature_data[feature_index].activation_hypothesis
        self.automated_interp_pipeline.predict_activation(data, hypothesis)

def sample_and_filter_data(
    df: pd.DataFrame, 
    quant_levels: int = 9, 
    final_sample_size: int = 10,
    samples_per_quant: int = 2,
    exceptions: Optional[Dict[int, int]] = None,
) -> tuple[Dict[Any, List[FeatureSample]], List[int]]:
    """
    Samples and filters data from the dataframe based on quantization levels and filters valid URLs.
    
    Args:
    df (pd.DataFrame): The dataframe to sample from.
    quant_levels (int): The number of quantization levels.
    final_sample_size (int): The number of random samples to take after processing all quantization levels.
    
    Returns:
    Tuple[Dict[int, List[dict]], List[int]]: A dictionary with quantization levels as keys and lists of sampled data as values,
                                              and a list of remaining indices after sampling.
    """
    sampled_data_by_quant : Dict[Any, List[FeatureSample]] = {}
    all_sampled_indices : List[int] = []

    for i in range(quant_levels):
        df_quantized = df[df['quantized_acts'] == i].sort_values(by='activation', ascending=False)

        if exceptions and i in exceptions:
            sample_size = min(exceptions[i], len(df_quantized))
        else:
            sample_size = min(samples_per_quant, len(df_quantized))
        valid_sampled_data, sampled_indices = sample_valid_data(df_quantized, sample_size)
        sampled_data_by_quant[i] = format(valid_sampled_data)
        all_sampled_indices.extend(sampled_indices)

    remaining_indices = list(set(df.index) - set(all_sampled_indices))
    if len(remaining_indices) > 0:
        random_samples = sample_valid_data(df.loc[remaining_indices], final_sample_size)[0]
        sampled_data_by_quant['random'] = format(random_samples)

    return sampled_data_by_quant, remaining_indices

def sample_valid_data(df: pd.DataFrame, sample_size: int) -> tuple[List[dict], List[int]]:
    """
    Samples valid data based on URL filtering from a dataframe.
    
    Args:
    df (pd.DataFrame): The dataframe to sample from.
    sample_size (int): The number of samples to attempt to take.
    
    Returns:
    Tuple[List[dict], List[int]]: A list of valid sampled data and a list of remaining indices after sampling.
    """
    valid_sampled_data = []
    remaining_indices : List[int] = df.index.tolist()
    sampled_indices : List[int] = []

    while len(valid_sampled_data) < sample_size and remaining_indices:
        sampled_indices = np.random.choice(
            remaining_indices, 
            size=min(sample_size, len(remaining_indices)), 
            replace=False
        ).tolist()

        sampled_data = df.loc[sampled_indices].to_dict(orient='records')
        valid_data = filter_valid_image_urls([row['url'] for row in sampled_data])
        filtered = [sampled_data[j] for j in range(len(sampled_data)) if valid_data[j]]
        valid_sampled_data.extend(filtered)

    return valid_sampled_data, list(set(sampled_indices))

def format(dataset : List[dict]) -> List[FeatureSample]:
    return [
        FeatureSample(
            quantized_activation=row['quantized_acts'],
            activation=row['activation'],
            content=ImageContent(image_url=row['url'], caption=row['caption'])
        ) for row in dataset if filter_valid_image_urls([row['url']])
    ]

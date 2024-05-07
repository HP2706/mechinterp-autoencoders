import os
import tqdm
import shutil
import pandas as pd
import numpy as np
import torch
from utils import get_device
from typing import List, Literal, Optional, Type, Union
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
    InterpretabilityData, 
    FeatureDescription, 
    FeatureSample,
    PipelineConfig
)
from automated_interpretability import AutomatedInterpretability
from utils import (
    write_models_to_json, 
    load_models_from_json,
    filter_non_zero_batch,
)
from litellm import completion
from Laion_Processing.dataloader import LaionDataset
import instructor
from modal import gpu, Secret, enter, method, build
from openai import OpenAI

torch.manual_seed(42)
np.random.seed(42)
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
            instructor.from_openai(OpenAI()), model=self.interpretability_model_name
            #instructor.from_litellm(completion, instructor.mode.Mode.MD_JSON), model=self.interpretability_model_name
        )
        #TODO is this a smart design decision?
        self.interp_save_path = os.path.join(self.save_path, f"interpretability_data_{self.model.dir_name}.parquet")
        if os.path.exists(self.interp_save_path):
            self.interp_df = pd.read_parquet(self.interp_save_path)
        else:
            self.interp_df = None

        self.feature_df_save_path = os.path.join(self.save_path, "features.json")
        if os.path.exists(self.feature_df_save_path):
            self.feature_data : List[FeatureDescription] = load_models_from_json(FeatureDescription, self.feature_df_save_path)
        else:
            self.feature_data = [] 



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
    def get_interpretability_correlation(
        self, 
        index: int,
        feature_or_neuron: Literal['feature', 'neuron'] = 'feature',
    ):
        df = self.get_activations_metadata(index).sort_values(by='activation', ascending=False)

        positive_samples = []
        negative_samples = []
        dataset = []

        # Process each quantization bin
        for i in range(9):
            sample_size = 10 if i == 8 else 2
            df_quantized = df[df['quantized_acts'] == i].sort_values(by='activation', ascending=False)
            valid_sampled_data = sample_valid_data(df_quantized, sample_size)

            #samples under 5 are considered negative samples
            #samples above 5 are considered positive samples
            if i > 5:
                positive_samples.extend(valid_sampled_data)
            else:
                negative_samples.extend(valid_sampled_data)

            dataset.extend(valid_sampled_data)
            if len(valid_sampled_data) < sample_size:
                print(f"Could not reach target sample size for quantized_acts level {i}. Expected {sample_size}, got {len(valid_sampled_data)}")

        # Sample additional random data
        remaining_indices = list(set(df.index) - set(sum([data.index.tolist() for data in dataset], [])))
        random_samples = df.loc[np.random.choice(remaining_indices, size=5, replace=False)]
        dataset.extend(random_samples.to_dict(orient='records'))

        # Process and format data
        formatted_data = format_data_for_interpretation(dataset)
        feature_hypothesis = self.automated_interp_pipeline.explain_activation(formatted_data)
        print("feature_hypothesis", feature_hypothesis)
        # Create and save feature description
        feature_description = FeatureDescription.build_feature_description(
            feature_hypothesis, 
            index, 
            feature_or_neuron, 
            positive_samples, 
            negative_samples
        )
        self.feature_data.append(feature_description)
        write_models_to_json(self.feature_data, self.feature_df_save_path)
        vol.commit()

def sample_valid_data(df, sample_size):
    valid_sampled_data = []
    remaining_indices = df.index.tolist()

    while len(valid_sampled_data) < sample_size and remaining_indices:
        sampled_indices = np.random.choice(remaining_indices, size=sample_size, replace=False)
        sampled_data = df.loc[sampled_indices].to_dict(orient='records')
        valid_data = filter_valid_image_urls([row['url'] for row in sampled_data])
        filtered = [sampled_data[j] for j in range(len(sampled_data)) if valid_data[j]]
        valid_sampled_data.extend(filtered)
        remaining_indices = list(set(remaining_indices) - set(sampled_indices))

    return valid_sampled_data

def format_data_for_interpretation(dataset):
    return [
        FeatureSample(
            quantized_activation=row['quantized_acts'],
            activation=row['activation'],
            content=ImageContent(image_url=row['url'], caption=row['caption'])
        ) for row in dataset if filter_valid_image_urls([row['url']])
    ]

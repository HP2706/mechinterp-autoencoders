import os
import tqdm
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
    ActivationExample, 
    ImageContent,
    InterpretabilityData, 
    FeatureDescription, 
    FeatureSample,
    LaionRowData,
    PipelineConfig
)
from automated_interpretability import AutomatedInterpretability
from utils import (
    write_models_to_json, 
    load_models_from_json,
)
from litellm import completion
from Laion_Processing.dataloader import LaionDataset
import instructor
from modal import gpu, Secret, enter, method, build
from openai import OpenAI
from utils import filter_non_zero

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

    def get_acts_dir(self, index : int) -> str:
        return f"{PATH}/laion_acts_idx_{index}_{self.model.dir_name}"

    def interpretability_data(self, index : int) -> Optional[List[pd.DataFrame]]:
        dirname = self.get_acts_dir(index)
        if not os.path.exists(dirname):
            return None
        return [
            pd.read_parquet(f"{dirname}/metadata_{i}.parquet") for i in range(len(os.listdir(dirname)))
        ]

    @method()
    def create_acts_dataset(
        self,
        index : int,
        n_files : int = 5,
    ) -> None:
        
        dirname = self.get_acts_dir(index)
        os.makedirs(dirname, exist_ok=True)

        for (i, (tensor, df_metadata)) in tqdm.tqdm(
            enumerate(self.dataset.iter_files(max_count=n_files)), 
            total=n_files,
            desc="Processing Files"
        ):            
            tensor = tensor.to(self.device)
            with torch.no_grad():
                batch = []
                removed_indices = []
                batch_size = 512*20*500 # note we are only using one weight idx, so 1000x less compute and memory
                for j in range(0, tensor.shape[0], batch_size):
                    result = self.model.get_single_feature_acts(tensor[j:j+batch_size], feature_index=index).cpu()
                    non_zero_indices, zero_indices = filter_non_zero(result)
                    removed_indices.extend(
                        (zero_indices + j).tolist() # get the global index
                    )
                    batch.append(result[non_zero_indices])
                
                filtered_acts = torch.cat(batch)

            df_metadata = df_metadata.drop(removed_indices)
            df_metadata['activation'] = filtered_acts
            #create quantized activations
            num_bins = 9
            df_metadata['quantized_acts'] = np.digitize(filtered_acts, bins = np.linspace(-1, 1, num_bins))
            df_metadata.to_parquet(f"{dirname}/metadata_{i}.parquet")
            vol.commit()


    @method()
    def get_interpretability_correlation(
        self, 
        index : int,
        feature_or_neuron : Literal['feature', 'neuron'] = 'feature',
    ):

        dataframes = self.interpretability_data(index)
        if dataframes is None:
            raise ValueError(f"Index {index} does not exist")
        
        print("number of dataframes", len(dataframes))
        df = pd.concat(dataframes) #this might not work for large dataframes
        df = df.sort_values(by='activation', ascending=False)
        print("df", df)
        print("columns", df.columns)
        print("len df", len(df))
        selected_indices = []
        dataset = []

        # for each bin, select 2 examples except for the last bin where we select 10 examples
        for i in range(9):
            sample_size = min(
                10 if i == 9 else 2, 
                len(df[df['quantized_acts'] == i])
            )
            sampled_indices = df[df['quantized_acts'] == i].sample(n=sample_size, random_state=42).index
            print("number of rows with quantized_acts level", i, len(df[df['quantized_acts'] == i]))
            selected_indices.extend(sampled_indices)
            dataset.extend(df.loc[sampled_indices].to_dict(orient='records')) # type: ignore

        #of remaining rows, select 5 random samples
        remaining_indices = list(set(df.index) - set(selected_indices))
        random_samples = df.loc[np.random.choice(remaining_indices, size=5, replace=False)]
        dataset.extend(random_samples.to_dict(orient='records')) # type: ignore

        formatted_data = [
            LaionRowData(
                image_url=row['url'],
                caption=row['caption'],
                quantized_activation=row['quantized_acts'],
            )
            for row in dataset
        ] 

        t0 = time.time()
        valid_data = filter_valid_image_urls([row.image_url for row in formatted_data])
        print("Time taken for filtering", time.time() - t0)
        print("len before filtering", len(formatted_data))
        formatted_data = [formatted_data[i] for i in range(len(formatted_data)) if valid_data[i]]
        print("len after filtering", len(formatted_data))
        print("Time taken for filtering", time.time() - t0)
        feature_hypothesis = self.automated_interp_pipeline.explain_activation(formatted_data[:3])

        print("feature_hypothesis", feature_hypothesis)
        feature_description = FeatureDescription(
            **feature_hypothesis.model_dump(),
            feature_or_neuron=feature_or_neuron,
            index=index,
            high_act_samples=[
                FeatureSample(
                    quantized_activation=row['quantized_acts'],
                    activation=row['activation'],
                    content=ImageContent(
                        image_url=row['url'],
                        caption=row['caption']
                    )
                ) for row in dataset
            ],
            low_act_samples=[
                FeatureSample(
                    quantized_activation=row['quantized_acts'],
                    activation=row['activation'],
                    content=ImageContent(
                        image_url=row['url'],
                        caption=row['caption']
                    )
                ) for row in dataset
            ],
        )
        return
        self.feature_data.append(feature_description)
        write_models_to_json(self.feature_data, self.feature_df_save_path)
        vol.commit()

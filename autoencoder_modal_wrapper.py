import tqdm
from utils import filter_non_zero_batch
from Laion_Processing.dataloader import LaionDataset
from typing import List
import pandas as pd
from autoencoder import AutoEncoderBase
import torch
from modal import method, gpu
from common import stub, vol, image, PATH, dataset_vol, LAION_DATASET_PATH

@stub.cls(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=60*60,
    _allow_background_volume_commits=True,
    gpu=gpu.A10G()    
)
class AutoEncoderWrapper:
    def __init__(self, checkpoint_path: str, dataset_kwargs: dict):
        self.model = AutoEncoderBase.load_from_checkpoint(checkpoint_path)
        self.dataset = LaionDataset(**dataset_kwargs)
   
    @method()
    def create_acts_dataset(
        self,
        n_files : int = 5,
    ) -> List[pd.DataFrame]:
        dataframes : List[pd.DataFrame] = []
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
                    scaled_batch = tensor[j:j+batch_size].to(self.model.cfg.device)
                    acts = self.model.forward(scaled_batch, 'with_acts')
                    non_zero_indices, _ = filter_non_zero_batch(acts, threshold=None)
                    
                    #if there are no non-zero activations, we skip the batch because 
                    # it is all zeros or below the activation threshold
                    if non_zero_indices.nelement() == 0:
                        print("no non-zero activations skipping")
                        continue
                    
                    # Get non-zero activations and their indices for the entire batch
                    non_zero_activations = acts[non_zero_indices]
                    non_zero_positions = (non_zero_activations != 0).nonzero(as_tuple=False)
                    original_indices = (non_zero_indices + j).tolist()
                    
                    # Extract the activation values using these indices
                    activation_values = non_zero_activations[non_zero_positions[:, 0], non_zero_positions[:, 1]]
                    print("activation values", activation_values.shape)
                    print("non zero positions", non_zero_positions.shape)
                    for idx, value in zip(non_zero_positions.tolist(), activation_values.tolist()):
                        df_rows.append(
                            {**df_metadata.iloc[original_indices[idx[0]]].to_dict(), 
                            'activation': value,
                            'feature_idx': idx[1],
                            'data_idx': original_indices[idx[0]]+nrows,
                        })
            if len(df_rows) > 0:
                dataframes.append(pd.DataFrame(df_rows))
            else:
                print("no rows in batch")
            nrows += len(df_metadata)
        
        print("dataframes modal autoencoder", dataframes)
        return dataframes


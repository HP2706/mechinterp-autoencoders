import tqdm
from utils import filter_non_zero_batch
from Laion_Processing.dataloader import LaionDataset
from typing import List
import pandas as pd
from autoencoder import AutoEncoderBase
import torch
from modal import method, gpu
from common import app, vol, image, PATH, dataset_vol, LAION_DATASET_PATH

@app.cls(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=60*60,
    _allow_background_volume_commits=True,
    gpu=gpu.A10G()    
)
class GpuPipeline:
    def __init__(self, checkpoint_path: str, dataset_kwargs: dict):
        self.model = AutoEncoderBase.load_from_checkpoint(checkpoint_path)
        if self.model.cfg.updated_anthropic_method:
            self.dataset = LaionDataset(**dataset_kwargs, d_hidden=self.model.W_dec.shape[0])
        else:
            self.dataset = LaionDataset(**dataset_kwargs)
   
    @method()
    def create_acts_dataset(
        self,
        n_files : int = 5,
    ) -> List[pd.DataFrame]:
        dataframes : List[pd.DataFrame] = []

        for ((tensor_path, tensor), (metadata_path, df_metadata)) in tqdm.tqdm(
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
                    non_zero_indices, _ = filter_non_zero_batch(acts, threshold=1e-3)
                    
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
                    for idx, value in zip(non_zero_positions.tolist(), activation_values.tolist()):
                        df_rows.append(
                            {
                                **df_metadata.iloc[original_indices[idx[0]]].to_dict(), 
                                'activation': value,
                                'feature_idx': idx[1],
                                'idx_in_file': original_indices[idx[0]],
                                'emb_path' : tensor_path,
                                'metadata_path' : metadata_path
                            }
                        )
            if len(df_rows) > 0:
                dataframes.append(pd.DataFrame(df_rows))
            else:
                print("no rows in batch")
        return dataframes
    
    @method()
    def get_cosine_sim(self, df : pd.DataFrame, sub_set : torch.Tensor)->tuple[float, float]:
        print("dataframe", df.head())

        import torch.nn.functional as F
        emb_tensor = torch.vstack(df['embedding'].tolist())
        mean_norm = torch.norm(emb_tensor, p=2, dim=1).mean()

        random_tensor = torch.randn(emb_tensor.shape) 
        #normalize to have mean norm as emb_tensor
        random_tensor = random_tensor / torch.norm(random_tensor, p=2, dim=1, keepdim=True) * mean_norm

        # Compute pairwise cosine similarity
        cosine_sim_matrix = F.cosine_similarity(emb_tensor.unsqueeze(1), emb_tensor.unsqueeze(0), dim=2)
        cosine_sim_random_matrix = F.cosine_similarity(emb_tensor.unsqueeze(1), random_tensor.unsqueeze(0), dim=2)
        cosine_sim_dataset_sub_matrix = F.cosine_similarity(sub_set.unsqueeze(1), emb_tensor.unsqueeze(0), dim=2)

        # Mask the diagonal (self-similarity)
        mask = torch.eye(cosine_sim_matrix.size(0), dtype=torch.bool)
        cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0)
        cosine_sim_random_matrix = cosine_sim_random_matrix.masked_fill(mask, 0)
        print("cosine sim matrix", cosine_sim_matrix.shape)
        print("cosine sim random matrix", cosine_sim_random_matrix.shape)
        # Compute the mean of the non-diagonal elements
        mean_cosine_sim = cosine_sim_matrix.sum() / (cosine_sim_matrix.size(0) * (cosine_sim_matrix.size(1) - 1))
        mean_cosine_sim_random = cosine_sim_random_matrix.sum() / (cosine_sim_random_matrix.size(0) * (cosine_sim_random_matrix.size(1) - 1))


        """ 
        #TODO FIX THIS
        import torch
        import torch.nn.functional as F

        a = torch.randn((100, 10))
        b = torch.randn((1000, 10))
        cosine_sim_matrix = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
        print(cosine_sim_matrix.shape)


        # Create a mask with the same shape as cosine_sim_matrix
        mask = torch.eye(cosine_sim_matrix.size(0), cosine_sim_matrix.size(1), dtype=torch.bool)
        cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0)
        mean_cosine_sim = cosine_sim_matrix.sum() / (cosine_sim_matrix.size(0) * (cosine_sim_matrix.size(1) - 1))
        print(mean_cosine_sim) """

        print("mean cosine sim between act and random: ", mean_cosine_sim_random)
        print("mean cosine sim between act and itself: ", mean_cosine_sim)
        return (mean_cosine_sim_random, mean_cosine_sim)




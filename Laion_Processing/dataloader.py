import os
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
import numpy as np
from common import LAION_DATASET_PATH, EMB_FOLDER, METADATA_FOLDER
import pandas as pd
from typing import Generator, Optional, List, Tuple, Union

class LaionDataset(Dataset):
    def __init__(
        self, 
        embeddings_path: str, 
        with_metadata: bool = False
    ):

        self.with_metadata = with_metadata
        self.vec_list = [os.path.join(embeddings_path, file) for file in os.listdir(embeddings_path) if file.endswith(".npy")]

        self.current_file_index = -1
        self.data = None
        self.metadata_df = None
        self.load_next_file()

    def load_next_file(self):
        self.current_file_index += 1
        if self.current_file_index < len(self.vec_list):
            self.data = torch.tensor(np.load(self.vec_list[self.current_file_index]))
        else:
            self.data = None

    def __len__(self):
        if self.data is not None:
            return self.data.shape[0]*len(self.vec_list)
        return 0

    def __getitem__(self, idx: int) -> Union[torch.Tensor, dict]:
        if self.data is None:
            self.load_next_file()
            if self.data is None:
                raise StopIteration("No more data to load")
        
        actual_idx = idx % len(self.data)
        return self.data[actual_idx]


class LaionFileLoader:
    def __init__(
        self, 
        batch_size: int, 
        embeddings_path: str = EMB_FOLDER, 
        metadata_path: str = METADATA_FOLDER,
        with_metadata: bool = False
    ):
        self.dataset = LaionDataset(
            embeddings_path=embeddings_path, 
            with_metadata=with_metadata
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )

    def __iter__(self):
        return iter(self.dataloader)
    
    def yield_batches(self) -> Generator[
        Union[torch.Tensor, Tuple[torch.Tensor, dict]], 
        None,
        None
    ]:
        for batch in self.dataloader:
            yield batch


def get_merged_df(
    metadata_path : str,
    emb_path: str,
    number : int
):
    metadata_files = [
        os.path.join(metadata_path, file) for file in os.listdir(metadata_path) if file.endswith(".parquet")
        if f"{number:04}" in file
    ]
    emb_files = [
        os.path.join(emb_path, file) for file in os.listdir(emb_path) if file.endswith(".npy")
        if f"{number:04}" in file
    ]
    assert len(metadata_files) == 1
    assert len(emb_files) == 1
    metadata_df = pd.read_parquet(metadata_files[0])
    emb = np.load(emb_files[0])
    assert metadata_df.shape[0] == emb.shape[0]
    metadata_df['embedding'] = emb.tolist() #this step take 50 seconds
    return metadata_df

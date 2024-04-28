import os
from altair import overload
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Generator, Optional, List, Tuple, Union

class LaionDataset(Dataset):
    def __init__(
        self, 
        emb_folder: str, 
        metadata_folder: Optional[str],
        return_tuple: bool
    ):
        
        emb_paths = [os.path.join(emb_folder, f) for f in os.listdir(emb_folder)]
        if len(emb_paths) == 0:
            raise ValueError("No embedding files found in emb_folder")
        assert all(emb_path.endswith(".npy") for emb_path in emb_paths), "All embedding files should be .npy"
        if return_tuple:
            if metadata_folder is None:
                raise ValueError("metadata_paths must be provided if return_tuple is True")
            metadata_paths = [os.path.join(metadata_folder, f) for f in os.listdir(metadata_folder)]
            self.metadata_paths = metadata_paths
            assert len(emb_paths) == len(metadata_paths), "Embedding and metadata files must be paired"
            assert all(metadata_path.endswith(".parquet") for metadata_path in metadata_paths), "All metadata files should be .parquet"
        

        self.emb_paths = emb_paths
        self.return_tuple = return_tuple
        self.current_file_index = -1
        self.data: Optional[torch.Tensor] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.load_next_file()

    def load_next_file(self):
        self.current_file_index += 1
        if self.current_file_index < len(self.emb_paths):
            self.data = torch.tensor(np.load(self.emb_paths[self.current_file_index]))
            if self.return_tuple:
                self.metadata_df = pd.read_parquet(self.metadata_paths[self.current_file_index])
        else:
            self.data = None
            self.metadata_df = None

    def __len__(self):
        if self.data is not None:
            return self.data.shape[0]
        return 0

        
        
    @overload
    def __getitem__(self, idx: int) -> torch.Tensor: ...
    @overload
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]: ...
    @overload
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, pd.DataFrame]: ...
    @overload
    def __getitem__(self, idx: slice) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[torch.Tensor, Tuple[torch.Tensor, dict], Tuple[torch.Tensor, pd.DataFrame]]:
        if self.data is None:
            self.load_next_file()
            if self.data is None:
                raise StopIteration("No more data to load")
        
        if isinstance(idx, int):
            if self.return_tuple:
                metadata_dict = self.metadata_df.iloc[idx % len(self.metadata_df)].to_dict() # type: ignore
                return self.data[idx % len(self.data)], metadata_dict
            else:
                return self.data[idx % len(self.data)]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.__len__())
            embeddings = []
            metadata_dicts = []
            while start < stop and self.data is not None:
                if start < len(self.data):
                    if self.return_tuple:
                        metadata_dict = self.metadata_df.iloc[start % len(self.metadata_df)].to_dict() # type: ignore
                        metadata_dicts.append(metadata_dict)
                        embeddings.append(self.data[start])
                    else:
                        embeddings.append(self.data[start])
                    start += step
                else:
                    self.load_next_file()
                    if self.data is None:
                        break  # Exit the loop if no more data is available
                    start, stop, step = idx.indices(self.__len__())
            if not embeddings:
                raise StopIteration("No more data to load")
            
            if self.return_tuple:
                return torch.stack(embeddings), pd.DataFrame(metadata_dicts)
            else:
                return torch.stack(embeddings)

class LaionFileLoader:
    def __init__(
        self, 
        batch_size: int, 
        emb_folder: str, 
    ):
        self.dataset = LaionDataset(
            emb_folder=emb_folder, 
            metadata_folder=None,
            return_tuple=False,
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )

    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        if self.dataset.data is not None:
            return self.dataset.data.shape[0] * len(self.dataset.emb_paths)
        return 0
    
    def yield_batches(self) -> Generator[
        Union[torch.Tensor, Tuple[torch.Tensor, dict]], 
        None,
        None
    ]:
        for batch in self.dataloader:
            yield batch
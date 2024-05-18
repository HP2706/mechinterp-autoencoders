import os
import torch
from typing import Any, overload
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Generator, Literal, Optional, List, Tuple, Union
from utils import load_tensor
from mechninterp_utils import scale_dataset

def check_inputs(kwargs):
    split = kwargs.get('split', None)
    if split is None:
        raise ValueError("split argument is required")
    else:
        if split not in ['train', 'test']:
            raise ValueError(f"split argument must be either 'train' or 'test' got {split}")

class LaionDataset(Dataset):
    def __init__(
        self, 
        emb_folder: str, 
        metadata_folder: Optional[str],
        split : Literal['train', 'test'],
        with_filenames : bool = False,
        train_share : float = 0.8,
        n_count : Optional[int] = None,
        return_tuple: bool = False,
        d_hidden : Optional[float] = None,
    ):
        check_inputs(locals())
        self.d_hidden = d_hidden
        self.with_filenames = with_filenames
        emb_paths = [os.path.join(emb_folder, f) for f in os.listdir(emb_folder)]
        if len(emb_paths) == 0:
            raise ValueError("No embedding files found in emb_folder")
        assert all(emb_path.endswith(".npy") for emb_path in emb_paths), "All embedding files should be .npy"
        
        metadata_paths = None
        if return_tuple:
            if metadata_folder is None:
                raise ValueError("metadata_paths must be provided if return_tuple is True")
            metadata_paths = [os.path.join(metadata_folder, f) for f in os.listdir(metadata_folder)]
            assert len(emb_paths) == len(metadata_paths), f"""
                Embedding and metadata files must be paired got different lengths 
                embs_path :{len(emb_paths)} {emb_paths}
                metadata_paths :{len(metadata_paths)} {metadata_paths}
            """
            assert all(metadata_path.endswith(".parquet") for metadata_path in metadata_paths), "All metadata files should be .parquet"
    
        if split == 'train':
            self.emb_paths = emb_paths[:int(len(emb_paths)*train_share)]
            if metadata_paths is not None:
                self.metadata_paths = metadata_paths[:int(len(metadata_paths)*train_share)]
        elif split == 'test':
            self.emb_paths = emb_paths[int(len(emb_paths)*train_share):]
            if metadata_paths is not None:
                self.metadata_paths = metadata_paths[int(len(metadata_paths)*train_share):]

        if n_count:
            self.emb_paths = self.emb_paths[:min(n_count, len(self.emb_paths))]
            if metadata_paths is not None:
                self.metadata_paths = self.metadata_paths[:min(n_count, len(self.metadata_paths))]

        self.return_tuple = return_tuple
        self.current_file_index = -1
        self.data: Optional[torch.Tensor] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.load_next_file()

    def iter_files(
        self, 
        max_count : Optional[int] = None
    )-> Generator[Tuple[torch.Tensor, pd.DataFrame], None, None]:
        count = 0
        for i in range(len(self.emb_paths)):
            try:
                tensors = load_tensor(self.emb_paths[i])
                if self.d_hidden:
                    tensors = scale_dataset(tensors, self.d_hidden)
                df = pd.read_parquet(self.metadata_paths[i])
                yield (tensors, df)
            except ValueError as e:
                print(e)
                print("file", self.emb_paths[i], "is not a valid numpy shape", "and ")
                print("accompanying parquet file", self.metadata_paths[i])
                continue
            count += 1
            if max_count is not None and count >= max_count:
                return
            

    @property
    def get_metadata(self) -> pd.DataFrame:
        if self.metadata_df is None:
            raise ValueError("metadata_df is not available")
        return self.metadata_df

    def get_data_by_idx(self, idx: int) -> Tuple[torch.Tensor, pd.DataFrame]:
        tensor = load_tensor(self.emb_paths[idx])
        metadata_df = pd.read_parquet(self.metadata_paths[idx])
        return tensor, metadata_df

    def load_next_file(self):
        self.current_file_index += 1
        if self.current_file_index < len(self.emb_paths):
            self.data = load_tensor(self.emb_paths[self.current_file_index])

            if self.d_hidden:
                self.data = scale_dataset(self.data, self.d_hidden)

            if self.return_tuple:
                self.metadata_df = pd.read_parquet(self.metadata_paths[self.current_file_index])
        else:
            self.data = None
            self.metadata_df = None

    def __len__(self):
        if self.data is not None:
            print("self.emb_paths len", len(self.emb_paths))
            return self.data.shape[0] * len(self.emb_paths)
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

class LaionDataLoader:
    def __init__(
        self, 
        batch_size: int, 
        emb_folder: str,
        split : Literal['train', 'test'],
        n_count : Optional[int] = None,
        d_hidden : Optional[float] = None,
        train_share : float = 0.8, 
    ):
        self.dataset = LaionDataset(
            emb_folder=emb_folder, 
            metadata_folder=None,
            return_tuple=False,
            split=split,
            train_share=train_share,
            n_count=n_count,
            d_hidden=d_hidden
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )


    def __iter__(self):
        return iter(self.dataloader)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        if isinstance(idx, int):
            return self.dataloader.dataset[idx]
        elif isinstance(idx, slice):
            return self.dataloader.dataset[idx]
    
    def __len__(self):
        if self.dataset.data is not None:
            number = (self.dataset.data.shape[0] * len(self.dataset.emb_paths)) / self.dataloader.batch_size # type: ignore
            return int(number)
        return 0
    
    def yield_batches(self) -> Generator[
        Union[torch.Tensor, Tuple[torch.Tensor, dict]], 
        None,
        None
    ]:
        for batch in self.dataloader:
            yield batch


def load_loaders(
    batch_size: int,
    emb_folder: str,
    train_share : float = 0.8,
    d_hidden : Optional[float] = None,
    n_counts : Tuple[Optional[int], Optional[int]] = (None, None)
) -> tuple[LaionDataLoader, LaionDataLoader]:
    '''
    Returns a tuple of (train loader, test loader)
    '''
    return (
        LaionDataLoader(
        batch_size=batch_size,
        emb_folder=emb_folder,
        split='train',
        train_share=train_share,
        d_hidden=d_hidden,
        n_count=n_counts[0]
        ),
        LaionDataLoader(
            batch_size=batch_size,
            emb_folder=emb_folder,
            split='test',
            train_share=train_share,
            d_hidden=d_hidden,
            n_count=n_counts[1]
        )
    )


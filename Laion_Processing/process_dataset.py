from datasets import load_dataset
from models import Transformer, TransformerConfig
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from datamodels import RunMetaData
from typing import List, Tuple, Union
from utils import get_model_memory_usage, lm_cross_entropy_loss, get_gpu_memory_usage
from common import app, PATH, vol, dataset_vol, LAION_DATASET_PATH,  image, EMB_FOLDER, METADATA_FOLDER
from modal import Image, Volume, gpu
import aiohttp
import asyncio

with image.imports():
    import os
    import torch
    from datasets import load_dataset, Dataset, load_from_disk
    from multiprocessing import Pool
    from tqdm import tqdm
    from functools import partial
    import pandas as pd
    import numpy as np
    import requests


def download_laion_file(dir_name: str, destination_folder: str)->bool:
    url = f"https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/{dir_name}"
    content = requests.get(url, allow_redirects=True).content

    open(
        os.path.join(destination_folder, dir_name.split("/")[-1]), 
        'wb'
    ).write(content)

    dataset_vol.commit()
    dataset_vol.reload()
    print(f"Downloaded {dir_name}")
    print("check if it is saved", os.listdir(destination_folder))
    return True

@app.function(
    image = image,
    volumes = { LAION_DATASET_PATH: dataset_vol},
    _allow_background_volume_commits = True,
    timeout=10*60, 
    concurrency_limit=20,
    container_idle_timeout=30
)
async def async_download_laion_file(
    dir_name: str, 
    destination_folder: str
)->Union[bool, Tuple[bool, Tuple[str, str]]]:
    
    url = f"https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/{dir_name}"
    try:    
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    file_path = os.path.join(destination_folder, dir_name.split("/")[-1])
                    with open(file_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    dataset_vol.commit()
                    dataset_vol.reload()
                    return True
                else:
                    print(f"Failed to download {dir_name}: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"Failed to download {dir_name}: {e}")
        return (False, (dir_name, destination_folder))
    
@app.function(
    image = image,
    volumes = { PATH: vol, LAION_DATASET_PATH: dataset_vol},
    _allow_background_volume_commits = True,
    timeout=60*60, 
)
def download_and_processs_laion_dataset():
    #two files are key the npy file consists of 938763 embeddings of 768 dimensions
    #metadata.parquet files 
    os.makedirs(LAION_DATASET_PATH, exist_ok=True)
    os.makedirs(EMB_FOLDER, exist_ok=True)
    os.makedirs(METADATA_FOLDER, exist_ok=True)
    indices = [i for i in range(0, 200)] # we have 200 chunks
    vector_names = [f"img_emb/img_emb_{i:04}.npy" for i in indices]
    metadata_names  = [f"laion2B-en-metadata/metadata_{i:04}.parquet" for i in indices]
    indices = [i for i in range(18,74)] # we have 200 chunks
    metadata_names  = [f"laion2B-en-metadata/metadata_{i:04}.parquet" for i in indices]
    
    lst = [
        *zip(vector_names,[EMB_FOLDER]*len(vector_names)), 
        *zip(metadata_names, [METADATA_FOLDER]*len(metadata_names))
    ]

    failed_lst = []
    for elm in tqdm(
        async_download_laion_file.starmap(lst),
    ):
        if isinstance(elm, tuple):
            print(f"Failed to download {elm[1]}")
            failed_lst.append(elm[1])
            continue
        print(elm)

    print(f"Failed to download {len(failed_lst)} files")
    dataset_vol.commit()
    


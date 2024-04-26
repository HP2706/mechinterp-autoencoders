""" from common import stub, PATH, vol, image
from autoencoder import AutoEncoder, AutoencoderConfig
from models import Transformer, TransformerConfig
from train_autoencoder import create_activations_dataset, train_autoencoder
from process_dataset import download_and_tokenize_dataset
from utils import load_activations

@stub.function(
    image = image,
    volumes={PATH: vol},        
)
def checkdir():
    print("testing create_activations_dataset works")
    import pandas as pd
    import os
    from datasets import load_dataset, load_from_disk, Dataset
    from train_autoencoder import model_name, DATASET_NAME
    from datamodels import ActivationMetaData
    path = f"{PATH}/activations_{model_name.replace('/', '-')}_{DATASET_NAME.replace('/', '_')}"
    files = os.listdir(path)
    print("files", files)
    for file in [file for file in files if file.endswith(".parquet")]:
        loaded_df = pd.read_parquet(f"{path}/{file}")
        a = load_activations(loaded_df, 512)

    

@stub.local_entrypoint()
def main():
    #download_and_tokenize_dataset.remote()
    #create_activations_dataset.remote()
    #checkdir.remote()
    train_autoencoder.remote()
"""


from mechinterp_pipeline import MechInterpPipeline, PipelineConfig

import pandas as pd
import torch
import tqdm


if __name__ == "__main__":
    from typing import List
    import numpy as np
    import time
    cfg = PipelineConfig(device="mps", batch_size = 1024)
    print(cfg)


    pipeline = MechInterpPipeline(
        model_name="gelu-1l",
        encoder_name="run1",
        dataset_name="NeelNanda/c4-code-20k",
        cfg=cfg,    
    )

    indices = np.random.randint(0, 500, 10).tolist()
    start = time.time()
    pipeline.build_and_interpret(indices, kwargs={'feature_or_neuron': 'feature'})
    print(f"time taken to build feature pipeline for {1} indices ", time.time() - start)

    start = time.time()
    pipeline.build_and_interpret(indices, kwargs={'feature_or_neuron': 'neuron'})
    print(f"time taken to build neuron pipeline for {len(indices)} indices ", time.time() - start)




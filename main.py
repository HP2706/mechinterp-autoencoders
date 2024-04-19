from common import stub, PATH, vol, image
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

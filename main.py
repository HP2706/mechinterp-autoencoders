from common import stub, PATH, vol, image
from autoencoder import AutoEncoder, AutoencoderConfig
from models import Transformer, TransformerConfig
from train_autoencoder import create_activations_dataset, BottleneckT5Autoencoder, yield_batches
from process_dataset import download_and_tokenize_dataset


@stub.function(
    image = image,
    volumes={PATH: vol},        
)
def checkdir():
    print("testing create_activations_dataset works")
    import pandas as pd
    import os
    from datasets import load_dataset, load_from_disk, Dataset
    print(os.listdir(PATH))
    from train_autoencoder import model_name, DATASET_NAME
    path = f"{PATH}/activations_{model_name.replace('/', '-')}_{DATASET_NAME.replace('/', '_')}"
    print(os.listdir(path))
    

@stub.local_entrypoint()
def main():
    #download_and_tokenize_dataset.remote()
    create_activations_dataset.remote()
    checkdir.remote()


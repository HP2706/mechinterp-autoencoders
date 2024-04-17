from common import stub, PATH, vol
from autoencoder import AutoEncoder, AutoencoderConfig
from models import Transformer, TransformerConfig
from train import train_model, download_dataset, image


@stub.function(
    image = image,
    volumes={PATH: vol},        
)
def checkdir():
    import os
    from datasets import load_dataset, load_from_disk
    DATASET_NAME = "roneneldan/TinyStories"
    for split in ['validation', 'train']:
        path = f"{PATH}/{DATASET_NAME}/{split}"
        dataset = load_from_disk(path)
        print(dataset.to_pandas().head(1))


@stub.local_entrypoint()
def main():
    #download_dataset.remote()
    #checkdir.remote()
    train_model.remote()



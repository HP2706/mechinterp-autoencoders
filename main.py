from common import stub, PATH, vol
from autoencoder import AutoEncoder, AutoencoderConfig
from models import Transformer, TransformerConfig
from train import train_model, download_dataset, image
import torch
cfg = AutoencoderConfig(
    seed=42, 
    batch_size=256, 
    buffer_mult=2, 
    lr=1e-3, 
    num_tokens=1000, 
    l1_coeff=1e-3, beta1=0.9, 
    beta2=0.999, dict_mult=100, seq_len=512, d_mlp=256, 
    enc_dtype="float32", remove_rare_dir=True, 
    model_batch_size=256, buffer_size=256, 
    buffer_batches=256, device='mps'
)

@stub.function(
    image = image,
    volumes={PATH: vol},        
)
def checkdir():
    import os
    print("os.listdir(PATH):", os.listdir(PATH))
    print(f"os.listdir('{PATH}/train'):", os.listdir(f"{PATH}/train"))
    print("GB size of PATH/train:", sum(os.path.getsize(f"{PATH}/train/{f}") for f in os.listdir(f"{PATH}/train")) / 1024**3)

@stub.local_entrypoint()
def main():
    checkdir.remote()



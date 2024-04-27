""" from common import stub, PATH, vol, image
from mechinterp_pipeline import MechInterpPipeline, PipelineConfig
from typing import List
import numpy as np
import time
from tqdm import tqdm

@stub.local_entrypoint()
def main():
    cfg = PipelineConfig(device="cuda", batch_size = 1024)
    print(cfg)

    pipeline = MechInterpPipeline(
        model_name="gelu-1l",
        encoder_name="run1",
        dataset_name="NeelNanda/c4-code-20k",
        cfg=cfg,    
    )

    indices =[101]#, 99, 18, 28]
    start = time.time()
    for x in tqdm(pipeline.build_and_interpret.starmap(
        zip(indices, [{'feature_or_neuron': 'feature'}]*len(indices)))):
        print(x)
    print(f"time taken to build feature pipeline for {1} indices ", time.time() - start)

    start = time.time()
    for x in tqdm(pipeline.build_and_interpret.starmap(
        zip(indices, [{'feature_or_neuron': 'neuron'}]*len(indices)))
    ):
        print(x)
    print(f"time taken to build neuron pipeline for {len(indices)} indices ", time.time() - start)
"""

from modal import method
from autoencoder import AutoEncoder, AutoencoderConfig, GatedAutoEncoder
import torch


d_mlp = 128
cfg = AutoencoderConfig(
    seed=42,
    batch_size=512*10,
    buffer_mult=10,
    lr=1e-3,
    l1_coeff=0.01,
    beta1=0.9,
    beta2=0.999,
    dict_mult=4,
    seq_len=512,
    d_mlp=d_mlp,
    buffer_size=1000000,
    buffer_batches=25,
    device="mps",
    n_epochs=10
)



vec = torch.randn(512, d_mlp)

gate_autoencoder = GatedAutoEncoder(cfg)
autoencoder = AutoEncoder(cfg)

loss_data =  gate_autoencoder.forward(vec, method="with_loss")

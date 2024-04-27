from common import stub, PATH, vol, image
from mechinterp_pipeline import MechInterpPipeline, PipelineConfig
from Laion_Processing.process_dataset import download_and_processs_laion_dataset
from train_clip_autoencoder import test_dataloader
from typing import List
import numpy as np
import time
from tqdm import tqdm

@stub.local_entrypoint()
def main():



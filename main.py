from autoencoder import AutoEncoder
from common import EMB_FOLDER, stub, PATH, vol, image, dataset_vol, LAION_DATASET_PATH, METADATA_FOLDER
from modal import gpu
from Laion_Processing.process_dataset import download_and_processs_laion_dataset
from Laion_Processing.dataloader import LaionDataset, LaionFileLoader
from clip_mechinterp_pipeline import ClipMechInterpPipeline
from train_clip_autoencoder import train_autoencoder, get_recons_loss
from typing import List, Type
import numpy as np
import os
import time
from tqdm import tqdm

@stub.function(
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},    
    image = image,
    #gpu=gpu.A10G(),
    timeout=10*60*60, #10 hours
)
def checkdir():
    print("os.listdir(PATH)", os.listdir(PATH))
    model_dir = f"{PATH}/laion2b_autoencoders"
    print("os.listdir(model_dir)", os.listdir(model_dir))
    for dir in os.listdir(model_dir):
        print("dir", dir)
        if os.path.isdir(f'{model_dir}/{dir}'):
            print(f"os.listdir('{model_dir}/{dir}')", os.listdir(f'{model_dir}/{dir}'))

@stub.function(
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},    
    image = image,
    #gpu=gpu.A10G(),
    timeout=10*60*60, #10 hours
) 
def run():
    model_path = f"{PATH}/laion2b_autoencoders"
    

@stub.local_entrypoint()
def main():
    #train_autoencoder.remote(2, 'autoencoder', 8, False)
    #checkdir.remote()
    model_path = f"{PATH}/laion2b_autoencoders"
    path = f'{model_path}/autoencoder_d_hidden_6144_lr_5e-05_dict_mult_8_epoch_0'
    pipeline = ClipMechInterpPipeline(
        path,
        interpretability_model_name='gpt-4-turbo', #"gemini/gemini-1.5-pro-latest",
        emb_folder=EMB_FOLDER,
        metadata_folder=METADATA_FOLDER,
        split='test',
        return_tuple=True,
        with_filenames=True,
    )
    #pipeline.check_activations.remote()https://modal.com/hp2706/storage/autoencoder/
    pipeline.create_acts_dataset.remote(40)
    #pipeline.get_interpretability_correlation.remote(1000)

""" 
import instructor
from openai import OpenAI
from automated_interpretability import AutomatedInterpretability
from datamodels import LaionRowData
automated_pipeline = AutomatedInterpretability(
    instructor.from_openai(OpenAI()), "gpt-4-turbo"
)

random_data = [
    LaionRowData(
        image_url="http://www.millersturf.com.au/wp-content/uploads/2017/01/leptosphaerulina-leaf-blight.jpg",
        caption="A cat sitting on a chair",
        quantized_activation=1,
    )
]

out = automated_pipeline.explain_activation(
    examples=random_data,
)
print(out) """
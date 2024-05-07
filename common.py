from modal import Stub, Volume, Image
stub = Stub(name="autoencoder anthropic")
vol = Volume.from_name("autoencoder", create_if_missing=True)
dataset_vol = Volume.from_name("laion_dataset", create_if_missing=True)
PATH = "/autoencoder"
LAION_DATASET_PATH = "/laion"
EMB_FOLDER =  f"{LAION_DATASET_PATH}/img_emb"
METADATA_FOLDER = f"{LAION_DATASET_PATH}/metadata"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-2-1-unclip-small"

MODELS_DIR = f"{PATH}/hf_models"
def download_model():
    from huggingface_hub import snapshot_download
    import os
    
    os.makedirs(MODELS_DIR, exist_ok=True)

    versions = ["small", "base", "large", "xl"]
    for version in [versions[0]]:
        MODEL_PATH = f"thesephist/contra-bottleneck-t5-{version}-wikipedia"
        snapshot_download(MODEL_PATH, local_dir=f"{MODELS_DIR}/{MODEL_PATH.replace('/', '-')}")


image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.11"
).pip_install_from_requirements(
    "requirements.txt"
)
#.run_function(download_model)


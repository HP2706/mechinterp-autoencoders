from modal import Stub, Volume, Image
stub = Stub(name="autoencoder anthropic")
vol = Volume.from_name("autoencoder anthropic", create_if_missing=True)
dataset_vol = Volume.from_name("laion_dataset", create_if_missing=True)
PATH = "/autoencoder"
DATASET_NAME = "roneneldan/TinyStories"
LAION_DATASET_PATH = "/laion"
EMB_FOLDER =  f"{LAION_DATASET_PATH}/img_emb"
METADATA_FOLDER = f"{LAION_DATASET_PATH}/metadata"

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
).pip_install(
    "torch",
    "transformer_lens",
    "transformers",
    "datasets",
    "einops",
    "pandas",
    "pydantic>=2.0",
    "huggingface_hub",
    "wandb",
    "tqdm",
    "pytest",
    "sentencepiece",
    "psutil",
    "instructor",
    "torchmetrics",
    "aiohttp",
    "pillow",
)
#.run_function(download_model)
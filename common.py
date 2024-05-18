from modal import Stub, Volume, Image, Secret

stub = Stub(
    name="autoencoder anthropic", 
    secrets=[
        Secret.from_name("my-gemini-secret"), 
        Secret.from_name("my-openai-secret"),
        Secret.from_name("my-logfire-secret"),
    ]
)

vol = Volume.from_name("autoencoder", create_if_missing=True)
dataset_vol = Volume.from_name("laion_dataset", create_if_missing=True)
PATH = "/autoencoder"
LAION_DATASET_PATH = "/laion"
EMB_FOLDER =  f"{LAION_DATASET_PATH}/img_emb"
METADATA_FOLDER = f"{LAION_DATASET_PATH}/metadata"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-2-1-unclip-small"
MODELS_DIR = f"{PATH}/hf_models"

def get_model_dir()-> str:
    return f"{PATH}/laion2b_autoencoders"

def check_processor_type():
    import platform
    machine_type = platform.machine()
    if machine_type.startswith('x86'):
        print("Running on an x86 processor.")
    elif machine_type.startswith('amd64') or machine_type == 'i386':
        print("Running on an AMD processor.")
    else:
        print("Unknown processor type.")


def download_java_run_time():
    print("downloading jdk")
    import jdk
    from jdk.enums import OperatingSystem, Architecture
    import os
    import platform
    check_processor_type()
    
    # Determine the architecture
    machine_type = platform.machine()
    if machine_type.startswith(('x86', 'amd64', 'i386')):
        print("Running on an x64 processor.")
        arch = Architecture.X64
    elif machine_type.startswith('arm'):
        print("Running on an ARM processor.")
        arch = Architecture.AARCH64
    else:
        print("Unknown processor type.")
        arch = None  # or handle the unknown case appropriately
    
    if arch:
        # Install the JDK
        jdk_path = jdk.install(
            version='17',
            operating_system=OperatingSystem.LINUX,
            arch=arch,
            vendor='Adoptium'
        )
        print("jdk path is", jdk_path)
        # Set the JAVA_HOME environment variable
        os.environ['JAVA_HOME'] = jdk_path

        print(f"JAVA_HOME set to {jdk_path} check:", os.getenv('JAVA_HOME'))
    else:
        print("Unsupported architecture.")

image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.11"
).pip_install_from_requirements(
    "requirements.txt"
).run_function(
    download_java_run_time
).env(
    {'JAVA_HOME': '/root/.jdk/jdk-17.0.11+9'} 
    #TODO find out why you need to set this manually and 
    #cannot do it with os.environ['JAVA_HOME'] = jdk_path
)


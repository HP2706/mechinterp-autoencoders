from common import (
    app, 
    vol, 
    image, 
    PATH, 
    LAION_DATASET_PATH, 
    EMB_FOLDER,
    dataset_vol,
)
from modal import gpu, Secret, Mount
import torch
import time
from Laion_Processing.dataloader import load_loaders

@app.function(
    image = image,
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},   
    timeout=10*60*60, #3 hours
    gpu=gpu.A10G(),    
    secrets=[Secret.from_name("my-wandb-secret")],
    _allow_background_volume_commits=True,
    mounts=[Mount.from_local_file("kernels.py", remote_path="/root/kernels.py")]
)
def benchmark():
    batch_size = 2048
    emb_dim = 768
    expansion_factor = 2

    from autoencoder import TopKAutoEncoder, TopKAutoEncoderConfig
    

    train_loader, test_loader = load_loaders(
        batch_size=(batch_size, 4*batch_size), 
        emb_folder=EMB_FOLDER,
        train_share=0.8,
        d_hidden = emb_dim * expansion_factor, 
        n_counts=(None, 5)
    )

    topk_cfg = TopKAutoEncoderConfig(
        type="topk_autoencoder", #TODO seems very stupid this variable
        dict_mult=expansion_factor,
        d_input=emb_dim,
        l1_coeff=0.0,
        k=10,
        k_aux=1,
    )

    model = TopKAutoEncoder(topk_cfg)
    device = "cuda"
    dtype = torch.float16
    model = model.to(device).to(dtype=dtype)
    ema_frequency_counter = torch.zeros((emb_dim * expansion_factor), device=device, dtype=dtype)

    def gen_sparse_vec(is_kernel : bool = False):
        sparsity_percentage = 0.8
        d_model = emb_dim * expansion_factor
        random_vector = torch.rand(batch_size, d_model, dtype=dtype)
        mask = torch.rand(batch_size, d_model, dtype=dtype) > sparsity_percentage
        random_vector[mask] = 0
        if is_kernel:
            values, indices = torch.topk(
                random_vector, 
                int((d_model * sparsity_percentage)*(1-sparsity_percentage)), 
                dim=1
            )
            return values, indices
        else:
            return random_vector


    for i, batch in enumerate(train_loader):
        print(f"Processing batch {i}")
        random_vector = gen_sparse_vec()
        random_vector = random_vector.to(device)
        t0 = time.time()
        model.decode(random_vector)
        print(f"NAIVE APPROACH: Time taken to process {batch_size} embeddings, with d_sae : {model.d_sae} : {time.time() - t0}")
        if i > 10:
            break

    for i, batch in enumerate(train_loader):
        print(f"Processing batch {i}")
        non_zero_values , non_zero_indices = gen_sparse_vec(True)
        non_zero_indices = non_zero_indices.to(device)
        non_zero_values = non_zero_values.to(device)
     
        t0 = time.time()
        print(non_zero_values.shape, non_zero_indices.shape)
        model.decode_kernel(non_zero_values, non_zero_indices)
        print(f"Custom Kernels: Time taken to process {batch_size} embeddings, with d_sae : {model.d_sae} : {time.time() - t0}")
        if i > 10:
            break


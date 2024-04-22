from utils import get_model_memory_usage, lm_cross_entropy_loss, get_gpu_memory_usage
from common import (
    stub, vol, image, 
    MODELS_DIR,PATH, DATASET_NAME
)
from autoencoder import AutoencoderConfig, AutoEncoder
from utils import load_activations
from modal import gpu
import modal
from datamodels import ActivationData, ActivationMetaData
import pandas as pd


MODEL_PATH = f"{MODELS_DIR}/thesephist-contra-bottleneck-t5-small-wikipedia"

with image.imports():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_from_disk
    from huggingface_hub import snapshot_download
    import torch
    from torch.optim import AdamW
    from typing import Any, Dict, List, Type, Union, Generator
    import wandb
    from tqdm import tqdm
    import psutil
    import time
    import os
    import json

version = "small"
model_name = f"thesephist/contra-bottleneck-t5-{version}-wikipedia"

@stub.cls(
    image = image,
    volumes={PATH: vol},    
    gpu=gpu.A10G(),
    concurrency_limit=100,   
    allow_concurrent_inputs=True,
    container_idle_timeout=40, # seconds
)
class BottleneckT5Autoencoder:
    @modal.build()
    def download_model(self):
        snapshot_download(model_name)

    @modal.enter()
    def load(self):
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        #TODO self.model = torch.compile(self.model) # we use torch.compile for potential speedup 
        # requires compiler installed 
        self.model.eval()

    @modal.method()
    async def get_embeddings(self, kwargs : Dict[str, torch.Tensor]) -> List[dict[str, Any]]:
        token_ids = kwargs['token_ids']
        attn_mask  = kwargs['attn_masks']

        embeddings = await self.embed(token_ids, attn_mask)
                
        return [
            ActivationData(text=None, activations=emb, token_ids=tokens).model_dump() 
            for (tokens, emb) in zip(token_ids.numpy().tolist(), embeddings.numpy().tolist())
        ]

    async def embed(self, token_ids: torch.Tensor, attn_mask : torch.Tensor) -> torch.Tensor:
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        token_ids = token_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)
        with torch.no_grad():
            reserved_memory_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            utilization_ratio = allocated_memory_gb / reserved_memory_gb if reserved_memory_gb > 0 else 0
            print(f"Memory reserved: {reserved_memory_gb:.2f} GB, Memory allocated: {allocated_memory_gb:.2f} GB, Utilization ratio: {utilization_ratio:.2f}")
            return self.model(
                input_ids=token_ids,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_inputs['input_ids'],
                encode_only=True,
            ).detach().cpu()
        
    @modal.method()
    async def embed_endpoint(self, texts: List[str]) -> torch.Tensor:
        '''a wrapper around embed that returns the embeddings as a tensor'''
        inputs = self.tokenizer(texts, return_tensors='pt',truncation=True, padding='max_length')
        return await self.embed(**inputs) #type: ignore
    
    @modal.method()
    @torch.no_grad()
    def generate_from_latent(self, latents: torch.Tensor, max_length=512, temperature=1.0) -> List[str]:
        '''
        Args:
            latents: a tensor of shape (N, D_model) where N is the number of texts and D_model is the dimension of the model
        Returns:
            List[str]: a list of strings of text generated from the latents
        '''
        batch_size = latents.shape[0] # N
        dummy_text = '.'
        tokens = self.tokenizer(dummy_text, return_tensors='pt')
        dummy_emb = self.embed(**tokens)  # This should be of shape [1, D_model] #type: ignore
        perturb_vector = latents - dummy_emb  # shape [batch_size, D_model]
       
        input_ids = self.tokenizer(
            [dummy_text] * batch_size, return_tensors='pt', padding=True
        ).to(self.device).input_ids #(batch_size, seq_len)
        perturb_vector = perturb_vector.unsqueeze(1).repeat(1, batch_size, input_ids.shape[1], 1).squeeze(0)
        self.model.perturb_vector = perturb_vector

        # Generate text from the model
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=batch_size,
        )
        tokens = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return tokens[:batch_size] # TODO this is a hacky workaround, and should be fixed in the model

def yield_batches(dataset, batch_size) -> Generator[dict, None, None]:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]

        yield {
            'token_ids': torch.tensor(list(batch['token_ids'])),
            'attn_masks': torch.tensor(list(batch['attn_masks']))
        } 

def save_accumulated_batches(accumulated_batches, activations_path, metadata, name):
    path = f"{activations_path}/v_{metadata.n_saved}_{name}.parquet"
    print("saving batches", len(accumulated_batches), accumulated_batches)
    df = pd.DataFrame(accumulated_batches)
    print("df", df.head())
    print("df activations", type(df['activations'][0]))
    df.to_parquet(path)  # Save DataFrame
    with open(f"{activations_path}/metadata_{name}", "w") as f:
        f.write(metadata.model_dump_json())
    vol.commit()

@stub.function(
    image = image,
    volumes={PATH: vol},   
    timeout=60*60, 
)
def create_activations_dataset():
    batch_size = 512
    model = BottleneckT5Autoencoder()
    
    activations_name = f"activations_{model_name.replace('/', '-')}_{DATASET_NAME.replace('/', '_')}"
    activations_path = f"{PATH}/{activations_name}"
    os.makedirs(activations_path, exist_ok=True)

    for name in ["train", "validation"]:
        DATASET_PATH = f"{PATH}/processed_{DATASET_NAME}"
        dataset : pd.DataFrame= load_from_disk(f"{DATASET_PATH}_{name}", keep_in_memory=False).to_pandas() # type: ignore
        print("dataset", name , dataset.head(), len(dataset))
        if not os.path.exists(f"{activations_path}/metadata_{name}"):
            metadata = ActivationMetaData(n_saved=0, last_idx=0)
        else:
            with open(f"{activations_path}/metadata_{name}", "r") as file:
                metadata_json = file.read()
            metadata = ActivationMetaData.model_validate_json(metadata_json)
        
        t0 = time.time()

        orig_size = len(dataset)
        
        accumulated_batches = []
        _interval_len = int(len(dataset) / batch_size)
        for (i, batch) in enumerate(model.get_embeddings.map(
                tqdm(yield_batches(dataset, batch_size), total=_interval_len), 
                return_exceptions=True
            )):
            if not isinstance(batch, Union[list, tuple]):
                print("exception", batch)
            else:
                accumulated_batches.extend(batch)
                
                memory = psutil.virtual_memory()
                used_memory_percentage = (memory.used / memory.total) * 100
                print(f"Used memory: {used_memory_percentage:.2f}% total memory: {memory.total / 1024**3:.2f} GB")
                metadata.last_idx = i*batch_size
                if used_memory_percentage >= 90:
                    save_accumulated_batches(accumulated_batches, activations_path, metadata, name)
                    accumulated_batches = []
                    metadata.n_saved += 1 
        
        metadata.last_idx = len(dataset)
        save_accumulated_batches(accumulated_batches, activations_path, metadata, name)
        

        time_taken = time.time() - t0
        print(f"Finished processing {name} dataset in {time_taken / 60} minutes for {orig_size} samples")


def save_model(model, cfg, model_path):
    torch.save(model.state_dict(), f"{model_path}/model.pth")
    with open(f"{model_path}/config.json", "w") as f:
        json.dump(cfg.dict(), f)

def load_model(model_path):
    with open(f"{model_path}/config.json", "r") as f:
        cfg = AutoencoderConfig.model_validate_json(f.read())
    model = AutoEncoder(cfg)
    model.load_state_dict(torch.load(f"{model_path}/model.pth"))
    return model, cfg

@stub.function(
    image = image,
    volumes={PATH: vol},   
    timeout=60*60, 
    gpu=gpu.A10G(),    
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def train_autoencoder():
    activations_name = f"activations_{model_name.replace('/', '-')}_{DATASET_NAME.replace('/', '_')}"

    if not os.path.exists(f"{PATH}/{activations_name}"):
        create_activations_dataset()

    model_dir = f"{PATH}/1L_autoencoder"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    d_mlp = 512 #TODO this should not be hardcoded
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
        remove_rare_dir=True,
        buffer_size=1000000,
        buffer_batches=25,
        device="cuda",
        n_epochs=10
    )

    model = AutoEncoder(cfg)
    wandb.init(
        # set the wandb project where this run will be logged
        project="Sparse AutoEncoder",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.lr,
            "architecture": "1Layer Transformer",
            "dataset": "TinyStories",
            "epochs": cfg.n_epochs,
        }
    )

    model.to(model.cfg.device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    for epoch in range(model.cfg.n_epochs):
        for train_file in [ file for file in os.listdir(f"{PATH}/{activations_name}") if "train" in file]:
            if not train_file.endswith(".parquet"):
                continue

            df = pd.read_parquet(f"{PATH}/{activations_name}/{train_file}")
            activations_batched = load_activations(df, model.cfg.batch_size)

            for activations in activations_batched:
                loss, x_reconstruct, acts, l2_loss, l1_loss = model.forward(
                    activations.to(model.cfg.device), method='with_loss'
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"epoch {epoch} train loss {loss.item()}")

        for eval_file in [file for file in os.listdir(f"{PATH}/{activations_name}") if "validation" in file]:
            if not eval_file.endswith(".parquet"):
                continue
            df = pd.read_parquet(f"{PATH}/{activations_name}/{eval_file}")
            activations_batched = load_activations(df, model.cfg.batch_size)
            for activations in activations_batched:
                with torch.no_grad():
                    loss, x_reconstruct, acts, l2_loss, l1_loss = model.forward(
                        activations.to(model.cfg.device), method='with_loss'
                    )
                    print(f"validation loss {loss.item()}")

    save_model(model, model.cfg, model_dir)
    print("model saved")
    print("checking if model can be loaded")
    model, cfg = load_model(model_dir)
    print("model loaded")

        
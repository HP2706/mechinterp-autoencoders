import asyncio
from threading import local
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from datamodels import ActivationData
from typing import Dict, List, Type, Union, Generator
from utils import get_model_memory_usage, lm_cross_entropy_loss, get_gpu_memory_usage
import torch
from common import (
    stub, vol, image, 
    MODELS_DIR,PATH, DATASET_NAME
)
from modal import gpu
import modal
from datamodels import ActivationData, ActivationMetaData
import pandas as pd


MODEL_PATH = f"{MODELS_DIR}/thesephist-contra-bottleneck-t5-small-wikipedia"

with image.imports():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_from_disk, Dataset
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    import pandas as pd
    import psutil
    import time
    import os
    import numpy as np
    import json

version = "small"
model_name = f"thesephist/contra-bottleneck-t5-{version}-wikipedia"

@stub.cls(
    image = image,
    volumes={PATH: vol},    
    gpu=gpu.A10G(),
    concurrency_limit=10,   
    allow_concurrent_inputs=True,
    container_idle_timeout=20, # seconds
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
        self.model.eval()

    @modal.method()
    async def get_embeddings(self, kwargs : Dict[str, torch.Tensor]) -> List[ActivationData]:
        token_ids = kwargs['token_ids']
        attn_mask  = kwargs['attn_masks']
        print(f"token_ids shape: {token_ids.shape}, attn_mask shape: {attn_mask.shape}")

        embeddings = await self.embed(token_ids, attn_mask)
        
        return [
            ActivationData(text=None, activations=emb, token_ids=tokens.tolist()) 
            for (tokens, emb) in zip(token_ids, embeddings.numpy())
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
        t0 = time.time()
        data= {
            'token_ids': torch.tensor(list(batch['token_ids'])),
            'attn_masks': torch.tensor(list(batch['attn_masks']))
        } 
        print(f"yielding batch of size {len(batch)} in {time.time() - t0} seconds")
        yield data

def save_accumulated_batches(accumulated_batches, activations_path, metadata, name):
    path = f"{activations_path}/v_{metadata.n_saved}_{name}.parquet"
    df = pd.DataFrame(accumulated_batches)
    df.to_parquet(path)  # Save DataFrame
    accumulated_batches.clear()  # Clear the list for new data
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
    print(f"Creating activations dataset at {activations_path}")

    for name in ["train", "validation"]:
        DATASET_PATH = f"{PATH}/processed_{DATASET_NAME}"
        dataset : pd.DataFrame= load_from_disk(f"{DATASET_PATH}_{name}", keep_in_memory=False).to_pandas() # type: ignore
        
        if not os.path.exists(f"{activations_path}/metadata_{name}"):
            metadata = ActivationMetaData(n_saved=0, last_idx=0)
        else:
            metadata = ActivationMetaData.model_validate_json(json.loads(open(file=f"{activations_path}/metadata_{name}").read()))



        lower = metadata.last_idx
        upper = len(dataset)
        print("upper", upper, "lower", lower)
        dataset_chunk = dataset[lower:upper]
        t0 = time.time()


        print(f"Processing {name} dataset with {len(dataset)} samples") #type: ignore
        orig_size = len(dataset)
        
        accumulated_batches = []
    
        _interval_len = int(len(dataset_chunk) / batch_size)
        for (i, batch) in enumerate(model.get_embeddings.map(
                tqdm(yield_batches(dataset_chunk, batch_size), total=_interval_len), 
                return_exceptions=True
            )):
            if not isinstance(batch, Union[list, tuple]):
                print("exception", batch)
            else:
                accumulated_batches.extend(batch)
                
                memory = psutil.virtual_memory()
                used_memory_percentage = (memory.used / memory.total) * 100
                metadata.last_idx = i*batch_size
                if used_memory_percentage >= 90:
                    save_accumulated_batches(accumulated_batches, activations_path, metadata, name)
                    metadata.n_saved += 1 
        
        metadata.last_idx = upper
        save_accumulated_batches(accumulated_batches, activations_path, metadata, name)
        

        time_taken = time.time() - t0
        print(f"Finished processing {name} dataset in {time_taken / 60} minutes for {orig_size} samples")


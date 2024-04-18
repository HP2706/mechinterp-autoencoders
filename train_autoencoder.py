import time
from datasets import load_dataset
from models import Transformer, TransformerConfig
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from datamodels import RunMetaData
from typing import List, Type
from utils import get_model_memory_usage, lm_cross_entropy_loss, get_gpu_memory_usage
import torch
from common import stub, PATH, vol
from modal import gpu
import modal
from datamodels import ActivationData

from train_transformer import image, DATASET_NAME


with image.imports():
    import os
    from transformers import AutoTokenizer
    from datasets import load_from_disk, Dataset
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from jaxtyping import Float, Int
    from transformer_lens.utils import download_file_from_hf  
    from transformer_lens.hook_points import HookPoint
    from transformer_lens import EasyTransformer, EasyTransformerConfig
    



@stub.cls(
    image = image,
    volumes={PATH: vol},    
    gpu=gpu.A10G(),
    concurrency_limit=10,   
)
class Model:
    @modal.build()
    def download(self):
        os.makedirs("SoLU_1L256W_C4_Width_Scan", exist_ok=True)
        download_file_from_hf("NeelNanda/SoLU_1L256W_C4_Width_Scan", "config.json", cache_dir="SoLU_1L256W_C4_Width_Scan")
        download_file_from_hf("NeelNanda/SoLU_1L256W_C4_Width_Scan", "model_final.pth", cache_dir="SoLU_1L256W_C4_Width_Scan")

    @modal.enter()
    def load(self):
        os.makedirs("SoLU_1L256W_C4_Width_Scan", exist_ok=True)
        json : dict = download_file_from_hf("NeelNanda/SoLU_1L256W_C4_Width_Scan", "config.json", cache_dir="SoLU_1L256W_C4_Width_Scan")#type: ignore
        state_dict : dict = download_file_from_hf("NeelNanda/SoLU_1L256W_C4_Width_Scan", "model_final.pth", cache_dir="SoLU_1L256W_C4_Width_Scan")#type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")

        config = EasyTransformerConfig(
            n_layers=json['n_layers'],
            d_model=json['d_model'],
            n_ctx=json['n_ctx'],
            d_head=json['d_head'],
            n_heads=json['n_heads'],
            d_mlp=json['d_mlp'],
            act_fn=json['act_fn'],
            d_vocab=json['d_vocab'],
            eps=json['ln_eps'],
            model_name='NeelNanda/SoLU_1L256W_C4_Width_Scan',
            tokenizer_name=json['tokenizer_name'],
            n_devices=1,
            normalization_type=json['normalization'],
            seed=json['seed'],
            init_weights=True,  # Assuming we want to initialize weights
            attn_only=json['attn_only'],
            n_params=json['n_params']  # Optional, provide default None if not present
        )
        model = EasyTransformer(config, move_to_device=False)#.load_state_dict(output)
        state_dict['unembed.b_U'] = torch.zeros(size=(json['d_vocab'],)) # we add an unembed bias of shape (d_vocab,) of zeros
        #since it is added, we do not change the model output, it is just so the keys match
        model.load_state_dict(state_dict)
        self.model = model

    @modal.method()
    def forward(self, inp : tuple[torch.Tensor, List[str]])->List[dict]:
        tokens, texts = inp
        tokens = tokens.to("cuda")
        datas = []
        print("forward called")

        def get_activation_hook(
            pattern: Float[torch.Tensor, "batch seq_len d_model"],
            hook: HookPoint,
        ):  
          
            print(f"hooked shape {pattern.shape} hook name {hook.name}")
            t0 = time.time()
            cloned_activation = pattern.clone().detach().cpu().numpy()  # Convert to NumPy array instead of list
            for text, act in zip(texts, cloned_activation):
                datas.append(ActivationData(text=text, activations=act).model_dump())

            print(f"hook time {time.time() - t0}")


        self.model.eval()
        self.model.remove_all_hook_fns() 

        out = self.model.run_with_hooks(
            tokens,
            return_type=None, # For efficiency, we don't need to calculate the logits
            fwd_hooks=[
                (
                    lambda name: name == "blocks.10.mlp.hook_post",
                    get_activation_hook
                ),
            ],   
        )
        print("output run with hooks", out)
        return datas

@stub.function(
    image = image,
    volumes={PATH: vol},   
    timeout=10*60, #5 minutes  
)
def create_activations_dataset():
    print("\n\nLoading datasets")
    DATASET_PATH = f"{PATH}/{DATASET_NAME}"
    train_set : Dataset = load_from_disk(f"{DATASET_PATH}/train", keep_in_memory=False) # type: ignore
    val_set : Dataset = load_from_disk(f"{DATASET_PATH}/validation", keep_in_memory=False) # type: ignore

    batch_size = 256

    print("\n\nCreating model")
    model = Model()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    for (name, dataset) in zip(["train", "val"], [train_set, val_set]):
        token_batches = []
        text_batches = []
        print(f"Processing {name} dataset with {len(dataset.to_pandas())} samples") #type: ignore
        
        for i in tqdm(range(0, len(dataset), batch_size)):
            if i > batch_size*100:
                break
            texts = dataset[i:i+batch_size]["text"]
            tokens = tokenizer.batch_encode_plus(
                texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt'
            ).input_ids
            token_batches.append(tokens)
            text_batches.append(texts)
            
            #yield dataset[i:i+batch_size]["text"]
        datas = list(model.forward.map(tqdm(zip(token_batches, text_batches)), return_exceptions=True))
        print("datas", datas)
        df = pd.DataFrame([elm for sublist in datas for elm in sublist])
        print(df.head(10))
        df.to_parquet(f"{PATH}/{name}_activations_10_mlp_hook_pre_{DATASET_NAME.replace('/', '_')}.parquet")

        vol.commit()


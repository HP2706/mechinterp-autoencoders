from typing import Union
import modal
from modal import gpu, method, build, enter
from common import (
    stub, PATH, vol, image, dataset_vol, LAION_DATASET_PATH, 
    METADATA_FOLDER, CLIP_MODEL_ID, DIFFUSION_MODEL_ID
)
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import AutoEncoder, AutoEncoderBase, GatedAutoEncoder



#these are copied from linus lee's great autoencoder repo https://github.com/thesephist/spectre
#specifically he found an interesting way to use gradient descent to optimize how we edit the latent space
@torch.enable_grad()
def dictgrad(
    spectre: Union[AutoEncoder, GatedAutoEncoder],
    x: torch.Tensor,
    f: torch.Tensor,
    original_features: torch.Tensor,
    # method-specific config
    steps: int = 500,
    **kwargs,
) -> torch.Tensor:
    """
    We perform gradient descent, initialized from the SAE reconstruction of the
    modified feature dictionary.
    """
    # "Reference" is the text, latent, and feature dictionary we want to edit.
    reference_features = original_features.clone().detach().requires_grad_(False)

    # Initialize with the "translate" edit method.
    latent = (
        translate(spectre, x, f, original_features, **kwargs)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # Adam optimizer with cosine annealing works faster than SGD, with minimal
    # loss in performance.
    optim = torch.optim.AdamW(
        [latent], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
    )
    optim.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, steps, eta_min=0, last_epoch=-1
    )

    # Gradient descent: we optimize MSE loss to our desired feature dictionary.
    for step in range(steps):
        features = spectre.forward(latent, method='with_acts')
        loss = F.mse_loss(features, reference_features)
        loss.backward()

        optim.step()
        optim.zero_grad()
        scheduler.step()
    return latent



def translate(
    spectre: Union[AutoEncoder, GatedAutoEncoder],
    x: torch.Tensor,
    f: torch.Tensor,
    original_features: torch.Tensor,
    # method-specific config
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    
    #currently only works for 1D tensors
    assert len(f.shape) == 1, "f should be a 1D tensor"
    assert len(original_features.shape) == 1, "original_features should be a 1D tensor"
    assert len(x.shape) == 1, "x should be a 1D tensor"

    x = x.clone()  # For in-place mutations in grad mode.
    for i in range(f.shape[0]):
        original_act = original_features[i]
        act = f[i]
        diff = act - original_act
        if diff.abs().item() > eps:
            #TODO is this correct for gated autoencoder?
            #note in the spectre implementation it is W_dec[:, i] this is because 
            # nn.Linear is used and it is transposed
            feature_vector = spectre.W_dec[i, :] 
            x += diff * feature_vector
    return x / x.norm()


@stub.cls(
    volumes={PATH: vol, LAION_DATASET_PATH: dataset_vol},
    image = image.pip_install('diffusers', 'accelerate' ),
    gpu=gpu.A10G(),    
)
class AutoEncoderPipeLine:
    @build() 
    def download_models(self):
        from huggingface_hub import snapshot_download
        snapshot_download(DIFFUSION_MODEL_ID)
        snapshot_download(CLIP_MODEL_ID)
    
    @enter()
    def load(self):
        from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
        tokenizer = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        text_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)

        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            DIFFUSION_MODEL_ID,
            torch_dtype=torch.float16,
            prior_tokenizer=tokenizer,
            prior_text_encoder=text_model,
        ).to("cuda")

        model_path = f"{PATH}/laion2b_autoencoders"
        self.autoencoder = AutoEncoderBase.load_from_checkpoint(
            f'{model_path}/autoencoder_d_hidden_6144_lr_5e-05_dict_mult_8_epoch_0'
        ).to("cuda")

    @method()
    @torch.no_grad()
    def gen_img(self, prompt: str, image_embedding: torch.Tensor):
        #image embedding of shape (batch, 768)
        return self.pipe( # type: ignore
            prompt=prompt, image_embeds=image_embedding.to("cuda").to(torch.float16) # type: ignore
        ).images

    @method()
    @torch.no_grad()
    def gen_img_spectre(self, feature_idx : int, strength : float, image_embedding: torch.Tensor):
        # we scale by 100 and normalize by sqrt(dict_mult*embedding_dim)
        # we scale clip embeddings to norm 100 for stability as is done in Data preprocessing 
        # in Laion_preprocessing/dataloader.py
        image_embedding = image_embedding * 100 
        emb_mean_norm = torch.mean(torch.norm(image_embedding, dim=-1, p=2))  # Compute L2 norm of each row
        desired_norm = torch.sqrt(torch.tensor(self.autoencoder.W_dec.shape[1]).float())
        scaling_factor = desired_norm / emb_mean_norm
        image_embedding = image_embedding * scaling_factor  # Scale the dataset
        image_embedding = image_embedding.to("cuda")

        feature_activations = self.autoencoder.forward(image_embedding.clone(), method='with_acts')

        edited_feature_activations = feature_activations.clone()
        edited_feature_activations[0, feature_idx] = strength

        edited_embedding = dictgrad(
            self.autoencoder,
            x=image_embedding[0],
            f=edited_feature_activations[0],
            original_features=feature_activations[0],
        ).unsqueeze(0) #(768)->(1, 768)

        return self.pipe(
            prompt="",
            image_embeds=edited_embedding.to("cuda").to(torch.float16)
        ).images

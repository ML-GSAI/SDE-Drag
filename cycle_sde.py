# *************************************************************************
# Copyright (2023) ML Group @ RUC
# 
# Copyright (2023) SDE-Drag Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def load_model(torch_device='cuda'):
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)
    return vae, tokenizer, text_encoder, unet, scheduler


@torch.no_grad()
def get_text_embed(prompt: list, tokenizer, text_encoder, torch_device='cuda'):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    return text_embeddings


@torch.no_grad()
def get_img_latent(img_path, vae, torch_device='cuda', dtype=torch.float32, height=None, weight=None):
    data = Image.open(img_path).convert('RGB')
    if height is not None:
        data = data.resize((weight, height))
    transform = transforms.ToTensor()
    data = transform(data).unsqueeze(0)
    data = (data * 2.) - 1.
    data = data.to(torch_device)
    data = data.to(dtype)
    latents = vae.encode(data).latent_dist.sample()
    latents = 0.18215 * latents
    return latents


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Sampler():
    def __init__(self, model, scheduler, num_steps=100):
        scheduler.set_timesteps(num_steps)
        self.num_inference_steps = num_steps
        self.num_train_timesteps = len(scheduler)

        self.alphas = scheduler.alphas
        self.alphas_cumprod = scheduler.alphas_cumprod

        self.final_alpha_cumprod = torch.tensor(1.0)
        self.initial_alpha_cumprod = torch.tensor(1.0)

        self.model = model

    @torch.no_grad()
    def get_eps(self, img, forward_img, timestep, guidance_scale, text_embeddings, lora_scale=None, is_diffedit=False):
        latent_model_input = torch.cat([img, img, forward_img, forward_img]) if guidance_scale > 1. else torch.cat(
            [img, forward_img])

        if not is_diffedit:
            text_embeddings = torch.cat([text_embeddings] * 2) if guidance_scale > 1. else torch.cat(
                [text_embeddings.chunk(2)[1]] * 2)
        else:
            text_embeddings = torch.cat([text_embeddings[0], text_embeddings[1]]) if guidance_scale > 1. else torch.cat(
                [text_embeddings[0].chunk(2)[1], text_embeddings[1].chunk(2)[1]])

        cross_attention_kwargs = None if lora_scale is None else {"scale": lora_scale}

        with torch.no_grad():
            noise_pred = self.model(latent_model_input, timestep, encoder_hidden_states=text_embeddings,
                                    cross_attention_kwargs=cross_attention_kwargs).sample

        if guidance_scale > 1.:
            noise_pred_uncond, noise_pred_text, forward_noise_pred_uncond, forward_noise_pred_text = noise_pred.chunk(4)
        elif guidance_scale == 1.:
            noise_pred_text, forward_noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond = 0.
            forward_noise_pred_uncond = 0.
        else:
            raise NotImplementedError(guidance_scale)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        forward_noise_pred = forward_noise_pred_uncond + guidance_scale * (
                forward_noise_pred_text - forward_noise_pred_uncond)

        return noise_pred, forward_noise_pred

    def sample(self, timestep, sample, forward_sample, forward_x_prev, guidance_scale, text_embeddings, sde=False,
               eta=1., lora_scale=None, is_diffedit=False):

        eps, forward_eps = self.get_eps(sample, forward_x_prev, timestep, guidance_scale, text_embeddings, lora_scale,
                                        is_diffedit)

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** (0.5) * (
                1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) if sde else 0

        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t_prev - sigma_t ** 2) ** (0.5)

        forward_alpha_prod_t = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.initial_alpha_cumprod
        forward_alpha_prod_t_prev = self.alphas_cumprod[timestep]

        forward_beta_prod_t_prev = 1 - forward_alpha_prod_t_prev

        forward_sigma_t_prev = eta * ((1 - forward_alpha_prod_t) / (1 - forward_alpha_prod_t_prev)) ** (0.5) * (
                1 - forward_alpha_prod_t_prev / forward_alpha_prod_t) ** (0.5)

        forward_pred_original_sample = (forward_x_prev - forward_beta_prod_t_prev ** (
            0.5) * forward_eps) / forward_alpha_prod_t_prev ** (0.5)
        forward_pred_sample_direction_coeff = (1 - forward_alpha_prod_t - forward_sigma_t_prev ** 2) ** (0.5)

        forward_noise = (forward_sample - forward_alpha_prod_t ** (
            0.5) * forward_pred_original_sample - forward_pred_sample_direction_coeff * forward_eps) / forward_sigma_t_prev

        img = alpha_prod_t_prev ** (
            0.5) * pred_original_sample + pred_sample_direction_coeff * eps + sigma_t * forward_noise

        return img

    def forward_sde(self, timestep, sample):
        prev_timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        x_prev = (alpha_prod_t_prev / alpha_prod_t) ** (0.5) * sample + (1 - alpha_prod_t_prev / alpha_prod_t) ** (
            0.5) * torch.randn_like(sample, device=sample.device)

        return x_prev

    def forward_ode(self, timestep, sample, guidance_scale, text_embeddings, lora_scale=None):
        prev_timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        eps = self.get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale)
        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * eps

        img = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help='random seed'
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of sampling steps"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.,
        help="classifier-free guidance scale"
    )
    parser.add_argument(
        "--float64",
        action='store_true',
        help="use double precision"
    )
    opt = parser.parse_args()
    set_seed(opt.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae, tokenizer, text_encoder, unet, scheduler = load_model(device)
    if opt.float64:
        torch.set_default_dtype(torch.float64)
        vae = vae.double()
        unet = unet.double()
        text_encoder = text_encoder.double()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.double()
    sampler = Sampler(model=unet, scheduler=scheduler, num_steps=opt.steps)

    prompt = ["a bowl of fruits"]
    guidance_scale = opt.scale

    # get text embedding
    text_embeddings = get_text_embed(prompt, tokenizer, text_encoder, device)
    uncond_embeddings = get_text_embed([""], tokenizer, text_encoder, device)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # get vae latent
    latents = get_img_latent('assets/origin.png', vae, device, dtype=vae.dtype)

    forward_sample = []
    forward_x_prev = []
    # forward process to get x_t0 and record and w'_s as Eq (7, 8)
    # t = [1, 2, ..., T-1]
    for t in tqdm(scheduler.timesteps[1:].flip(dims=[0]), desc="SDE Forward"):
        forward_sample.append(latents.clone())
        latents = sampler.forward_sde(t, latents)
        forward_x_prev.append(latents.clone())

    # cycle_sde sampling as Eq (6)
    # t = [T, T-1, ..., 2]
    for t in tqdm(scheduler.timesteps[:-1], desc="SDE Backward"):
        latents = sampler.sample(t, latents, forward_sample.pop(), forward_x_prev.pop(), guidance_scale,
                                 text_embeddings, sde=True)

    # VAE decode
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)

    path = f'output/cycle_sde_reconstruction'
    os.makedirs(path, exist_ok=True)
    save_image(image, os.path.join(path, f'step={opt.steps}-cfg={opt.scale}-float64={opt.float64}.png'))


if __name__ == "__main__":
    main()

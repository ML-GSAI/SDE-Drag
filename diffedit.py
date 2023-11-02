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

import torch
from diffusers import (DDIMInverseScheduler, DDIMScheduler,
                       StableDiffusionDiffEditPipeline)
from PIL import Image
from tqdm.auto import tqdm

from cycle_sde import Sampler, get_img_latent, get_text_embed, set_seed
from torchvision.utils import save_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help='random seed'
    )
    parser.add_argument(
            "--img_path",
            type=str,
            default='assets/origin.png',
            help='image path'
    )
    parser.add_argument(
            "--source_prompt",
            type=str,
            default='a bowl of fruits',
            help='prompt of source image'
    )
    parser.add_argument(
            "--target_prompt",
            type=str,
            default='a bowl of bananas',
            help='prompt of target image'
    )
    parser.add_argument(
            "--steps",
            type=int,
            default=50,
            help="discretize [0, T] into 50 steps"
        )
    parser.add_argument(
            "--scale",
            type=float,
            default=7.5,
            help="classifier-free guidance scale"
        )
    parser.add_argument(
            "--encode_ratio",
            type=float,
            default=0.7,
            help="encode ratio"
    )
    parser.add_argument(
            "--sde",
            action='store_true',
            help="use diffedit-sde",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = get_args()
    set_seed(opt.seed)

    pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()

    raw_image = Image.open(opt.img_path)
    source_prompt = opt.source_prompt
    target_prompt = opt.target_prompt

    # We use the default DiffEdit pipeline to generate mask.
    # Whether it's DiffEdit-SDE or DiffEdit-ODE, both use the same mask.
    mask_image = pipeline.generate_mask(
        image=raw_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )

    path = f'output/diffedit'
    os.makedirs(path, exist_ok=True)
    if opt.sde:
        t_0 = int(opt.encode_ratio * opt.steps)
        t_begin = opt.steps - t_0

        vae, tokenizer, text_encoder, unet, scheduler = pipeline.vae, pipeline.tokenizer, pipeline.text_encoder, pipeline.unet, pipeline.scheduler
        sampler = Sampler(model=unet, scheduler=scheduler, num_steps=opt.steps)

        # get text embeding
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text_embeddings_origin = get_text_embed([source_prompt], tokenizer, text_encoder, device)
        text_embeddings_edit = get_text_embed([target_prompt], tokenizer, text_encoder, device)
        uncond_embeddings = get_text_embed([""], tokenizer, text_encoder, device)

        text_embeddings_edit = torch.cat([uncond_embeddings, text_embeddings_edit])
        text_embeddings_origin = torch.cat([uncond_embeddings, text_embeddings_origin])

        # get VAE latent
        latents = get_img_latent(opt.img_path, vae, device, text_embeddings_origin.dtype)

        noises = []
        imgs = []
        imgs.append(latents)
        # forward process to get x_t0 and record and w'_s as Eq (7, 8)
        for t in tqdm(scheduler.timesteps[t_begin:].flip(dims=[0]), desc="SDE Forward"):
            latents, noise = sampler.forward_sde(t, latents, opt.scale, text_embeddings_origin)
            noises.append(noise)
            imgs.append(latents)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image).to(vae.device)
        imgs.pop()

        # cycle_sde sampling as Eq (6)
        for t in tqdm(scheduler.timesteps[t_begin - 1:-1], desc="SDE Backward"):
            latents = sampler.sample(t, latents, opt.scale, text_embeddings_edit, sde=True, noise=noises.pop())
            image_latents = imgs.pop()
            latents = latents * mask_image + image_latents * (1 - mask_image)

        # VAE decode
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        save_image(image, os.path.join(path, 'diffedit-sde.png'))
    else:
        # We followed the guidelines at https://huggingface.co/docs/diffusers/main/en/using-diffusers/diffedit
        # for the implementation of DiffEdit-ODE.
        inv_latents = pipeline.invert(
            prompt=source_prompt,
            image=raw_image,
            guidance_scale=opt.scale,
            inpaint_strength=opt.encode_ratio,
            num_inference_steps=opt.steps,
        ).latents

        image = pipeline(
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            negative_prompt=source_prompt,
            guidance_scale=opt.scale,
            inpaint_strength=opt.encode_ratio,
            num_inference_steps=opt.steps,
        ).images[0]

        image.save(os.path.join(path, 'diffedit-ode.png'))


if __name__ == "__main__":
    main()

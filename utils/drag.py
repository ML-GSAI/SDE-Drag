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

import math
import os

import PIL
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm

from cycle_sde import (Sampler, get_img_latent, get_text_embed, load_model, set_seed)


def scale_schedule(begin, end, n, length, type='linear'):
    if type == 'constant':
        return end
    elif type == 'linear':
        return begin + (end - begin) * n / length
    elif type == 'cos':
        factor = (1 - math.cos(n * math.pi / length)) / 2
        return (1 - factor) * begin + factor * end
    else:
        raise NotImplementedError(type)


def forward(img, scheduler, sampler, steps, drag_t, lora_scale_min, text_embeddings):
    forward_sample = []
    forward_x_prev = []
    lora_scales = []
    cfg_scales = []

    latents = img
    drag_t = int(drag_t * steps)
    t_begin = steps - drag_t
    length = len(scheduler.timesteps[t_begin - 1:-1]) - 1

    index = 1

    forward_x_prev.append(latents)
    for t in tqdm(scheduler.timesteps[t_begin:].flip(dims=[0]), desc="Forward"):
        lora_scale = scale_schedule(1, lora_scale_min, index, length, type='cos')
        cfg_scale = scale_schedule(1, 3, index, length, type='linear')

        forward_sample.append(latents.clone())
        latents = sampler.forward_sde(t, latents)
        forward_x_prev.append(latents.clone())

        lora_scales.append(lora_scale)
        cfg_scales.append(cfg_scale)
        index += 1
    return latents.clone(), forward_sample, forward_x_prev, lora_scales, cfg_scales


def backward(scheduler, sampler, mask, steps, drag_t, latents, forward_sample, forward_x_prev, lora_scales, cfg_scales,
             text_embeddings):
    drag_t = int(drag_t * steps)
    t_begin = steps - drag_t

    for t in tqdm(scheduler.timesteps[t_begin - 1:-1], desc="Backward"):
        latents = sampler.sample(t, latents, forward_sample.pop(), forward_x_prev.pop(), cfg_scales.pop(),
                                 text_embeddings, sde=True, lora_scale=lora_scales.pop())
        latents = torch.where(mask > 128, latents, forward_x_prev[-1])
    return latents.clone()


def copy_and_paste(latents, source_new, target_new, r, max_height, max_width, img_scale, noise_scale, device):
    def adaption_r(source, target, r, max_height, max_width):
        r_x_lower = min(r, source[0], target[0])
        r_x_upper = min(r, max_width - source[0], max_width - target[0])
        r_y_lower = min(r, source[1], target[1])
        r_y_upper = min(r, max_height - source[1], max_height - target[1])
        return r_x_lower, r_x_upper, r_y_lower, r_y_upper

    for source_, target_ in zip(source_new, target_new):
        r_x_lower, r_x_upper, r_y_lower, r_y_upper = adaption_r(source_, target_, r, max_height, max_width)

        source_feature = \
            latents[:, :, source_[1] - r_y_lower: source_[1] + r_y_upper,
            source_[0] - r_x_lower: source_[0] + r_x_upper].clone()

        latents[:, :, source_[1] - r_y_lower: source_[1] + r_y_upper, source_[0] - r_x_lower: source_[0] + r_x_upper] = \
            img_scale * source_feature + noise_scale * torch.randn(latents.shape[0], 4, r_y_lower + r_y_upper,
                                                                   r_x_lower + r_x_upper, device=device)

        latents[:, :, target_[1] - r_y_lower: target_[1] + r_y_upper,
        target_[0] - r_x_lower: target_[0] + r_x_upper] = source_feature * 1.1
    return latents


def drag(img_path, mask_path, prompt, source, target, drag_t, steps, step_size, img_scale, adapt_r, use_lora,
         lora_scale_min, save_lora_path, save_path, seed):
    set_seed(seed)

    noise_scale = (1 - img_scale ** 2) ** (0.5)

    vae, tokenizer, text_encoder, unet, scheduler = load_model()
    sampler = Sampler(model=unet, scheduler=scheduler, num_steps=steps)

    device = 'cuda'

    source = torch.tensor(source).div(torch.tensor([8]), rounding_mode='trunc')
    target = torch.tensor(target).div(torch.tensor([8]), rounding_mode='trunc')

    d = target - source
    d_norm_max = torch.norm(d.float(), dim=1, keepdim=True).max()

    if d_norm_max <= step_size:
        drag_num = 1
    else:
        drag_num = d_norm_max.div(torch.tensor([step_size]), rounding_mode='trunc')
        if (d_norm_max / drag_num - step_size).abs() > (d_norm_max / (drag_num + 1) - step_size).abs():
            drag_num += 1

    if use_lora:
        with torch.autocast(device):
            img = get_img_latent(img_path, vae)

            unet.load_attn_procs(save_lora_path)

            text_embeddings = get_text_embed(prompt, tokenizer, text_encoder)
            uncond_embeddings = get_text_embed([""], tokenizer, text_encoder)
    else:
        img = get_img_latent(img_path, vae)

        text_embeddings = get_text_embed(prompt, tokenizer, text_encoder)
        uncond_embeddings = get_text_embed([""], tokenizer, text_encoder)

    mask = PIL.Image.open(mask_path).resize((img.shape[3], img.shape[2]))
    mask = torch.tensor(np.array(mask))
    mask = mask.unsqueeze(0).expand_as(img).to(device)

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    for i in range(int(drag_num)):
        source_new = source + (i / drag_num * d).to(torch.int)
        target_new = source + ((i + 1) / drag_num * d).to(torch.int)

        if use_lora:
            with torch.autocast(device):
                latents, forward_sample, forward_x_prev, lora_scales, cfg_scales = forward(img, scheduler, sampler,
                                                                                           steps, drag_t,
                                                                                           lora_scale_min,
                                                                                           text_embeddings)
                latents = copy_and_paste(latents, source_new, target_new, adapt_r, img.shape[2] - 1, img.shape[3] - 1,
                                         img_scale, noise_scale, device)
                img = backward(scheduler, sampler, mask, steps, drag_t, latents, forward_sample, forward_x_prev,
                               lora_scales, cfg_scales, text_embeddings)
        else:
            latents, forward_sample, forward_x_prev, lora_scales, cfg_scales = forward(img, scheduler, sampler, steps,
                                                                                       drag_t, lora_scale_min,
                                                                                       text_embeddings)
            latents = copy_and_paste(latents, source_new, target_new, adapt_r, img.shape[2] - 1, img.shape[3] - 1,
                                     img_scale, noise_scale, device)
            img = backward(scheduler, sampler, mask, steps, drag_t, latents, forward_sample, forward_x_prev,
                           lora_scales, cfg_scales, text_embeddings)

        # Scale and decode the image latents with vae

        current_img = 1 / 0.18215 * img
        with torch.no_grad():
            if use_lora:
                with torch.autocast(device):
                    current_img = vae.decode(current_img).sample
            else:
                current_img = vae.decode(current_img).sample

        current_img = (current_img / 2 + 0.5).clamp(0, 1)

        if i == int(drag_num) - 1:
            save_image(current_img, os.path.join("output", save_path, "out_image_final.png"))
        else:
            save_image(current_img, os.path.join("output", save_path, "out_image" + str(i) + ".png"))

        current_img = current_img.cpu().permute(0, 2, 3, 1).numpy()[0]
        current_img = (current_img * 255).astype(np.uint8)

        if i == int(drag_num) - 1:
            yield "Drag Finish.", current_img
        else:
            yield "Dragging " + str(i + 1) + " / " + str(int(drag_num)) + " steps.", current_img

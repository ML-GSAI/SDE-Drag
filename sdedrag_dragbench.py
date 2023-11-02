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

import torch
import os
import argparse
import tempfile
import numpy as np
from tqdm.auto import tqdm
import PIL
import math

from PIL import Image
from torchvision.utils import  save_image
from cycle_sde import Sampler, load_model, get_img_latent, get_text_embed, set_seed
import json

from utils.train_lora import train_lora

def load_data(root='DragBench'):
    '''
    load data from DragBench
    '''
    files = [str(i) for i in range(17)]

    datas = {}
    for path, _, _ in os.walk(root):
        if any(file in path for file in files):
            data = {}
            img = os.path.join(path, 'origin_image.png')
            mask_path = os.path.join(path, 'mask.png')

            with open(os.path.join(path, 'prompt.json'), 'r') as f:
                prompt = json.load(f)
            data['source'] = prompt['source']
            data['target'] = prompt['target']
            data['prompt'] = prompt['prompt']
            data['mask'] = mask_path
            datas[img] = data

    return datas


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


def forward(opt, latents):
    '''
    noise adding process and analyticly compute \bar{w}
    '''
    noises = []
    imgs = []
    lora_scales = []
    cfg_scales = []

    imgs.append(img)
    t_0 = int(opt.t_0 * opt.steps)
    t_begin = opt.steps - t_0
    length = len(scheduler.timesteps[t_begin - 1:-1]) - 1

    index = 1
    for t in tqdm(scheduler.timesteps[t_begin:].flip(dims=[0]), desc="Forward"):
        lora_scale = scale_schedule(1, opt.lora_scale_min, index, length, type='cos')
        cfg_scale = scale_schedule(1, opt.scale, index, length, type='linear')
        latents, noise = sampler.forward_sde(t, latents, cfg_scale, text_embeddings, lora_scale=lora_scale)
        noises.append(noise)
        imgs.append(latents)
        lora_scales.append(lora_scale)
        cfg_scales.append(cfg_scale)
        index += 1
    return latents, noises, imgs, lora_scales, cfg_scales


def backward(opt, latents, noises, hook_latents, lora_scales, cfg_scales):
    '''
    SDE sampling with analyticly computed \bar{w}
    '''
    t_0 = int(opt.t_0 * opt.steps)
    t_begin = opt.steps - t_0

    hook_latent = hook_latents.pop()
    latents = torch.where(mask > 128, latents, hook_latent)
    for t in tqdm(scheduler.timesteps[t_begin - 1:-1], desc="Backward"):
        latents = sampler.sample(t, latents, cfg_scales.pop(), text_embeddings, sde=True, noise=noises.pop(), lora_scale=lora_scales.pop())
        hook_latent = hook_latents.pop()
        latents = torch.where(mask > 128, latents, hook_latent)
    return latents


def train_lora_and_load(image_path, prompt, unet):
    image = np.array(Image.open(image_path))
    model_path = 'runwayml/stable-diffusion-v1-5'
    lora_step = opt.lora_steps
    lora_rank = 4

    with tempfile.TemporaryDirectory() as temp_path:
        save_lora_path = temp_path
        train_lora(image, prompt, model_path, save_lora_path, output_path=None, lora_step=lora_step, lora_rank=lora_rank)
        unet.load_attn_procs(save_lora_path)


def get_args():
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
            default=200,
            help="discretize [0, T] into 200 steps"
        )
    parser.add_argument(
            "--t_0",
            type=float,
            default=0.6,
            help="copy and paste at x_{t_0}"
        )
    parser.add_argument(
            "--scale",
            type=float,
            default=3.,
            help="CFG scale"
        )
    parser.add_argument(
            "--step_size",
            type=float,
            default=2.,
            help="drag by 2 pixels towards the target point each time"
        )
    parser.add_argument(
            "--r",
            type=int,
            default=5,
            help="radius of the square area for copy and paste."
        )
    parser.add_argument(
            "--beta",
            type=float,
            default=0.3,
            help="the noise level added to source area"
    )
    parser.add_argument(
            "--alpha",
            type=float,
            default=1.1,
            help="Enhance the signal strength of the target point."
    )
    parser.add_argument(
            "--lora_steps",
            type=int,
            default=100,
            help="LoRA finetuning steps"
    )
    parser.add_argument(
            "--lora_scale_min",
            type=float,
            default=0.5,
            help="reduce the LoRA scale from 1 to 0.5 as time goes from 0 to t_0"
    )
    opt = parser.parse_args()
    return opt


def copy_and_paste(latents, source_new, target_new):

    def adaption_r(source, target, r, max_val=63):
        '''
        Adjust r to prevent arrays from going out of bounds
        '''
        r_x_lower = min(r, source[0], target[0])
        r_x_upper = min(r, max_val - source[0], max_val - target[0])
        r_y_lower = min(r, source[1], target[1])
        r_y_upper = min(r, max_val - source[1], max_val - target[1])
        return r_x_lower, r_x_upper, r_y_lower, r_y_upper

    for source_, target_ in zip(source_new, target_new):
        r_x_lower, r_x_upper, r_y_lower, r_y_upper = adaption_r(source_, target_, opt.r)

        source_feature = \
            latents[:, :, source_[1] - r_y_lower: source_[1] + r_y_upper,
            source_[0] - r_x_lower: source_[0] + r_x_upper].clone()

        latents[:, :, source_[1] - r_y_lower: source_[1] + r_y_upper, source_[0] - r_x_lower: source_[0] + r_x_upper] = \
            opt.beta * source_feature + noise_scale * torch.randn(latents.shape[0], 4, r_y_lower + r_y_upper, r_x_lower + r_x_upper, device=torch_device)

        latents[:, :, target_[1] - r_y_lower: target_[1] + r_y_upper, target_[0] - r_x_lower: target_[0] + r_x_upper] = source_feature * opt.alpha
    return latents


opt = get_args()
set_seed(opt.seed)
noise_scale = (1 - opt.beta ** 2) ** (0.5)

vae, tokenizer, text_encoder, unet, scheduler = load_model()
sampler = Sampler(model=unet, scheduler=scheduler, num_steps=opt.steps)

torch_device = 'cuda'

data = load_data()
for path, item in data.items():
    prompt = item['prompt']
    source = torch.tensor(item['source']).div(torch.tensor([8]), rounding_mode='trunc')
    target = torch.tensor(item['target']).div(torch.tensor([8]), rounding_mode='trunc')
    mask = PIL.Image.open(item['mask']).resize((64, 64))
    mask = torch.tensor(np.array(mask))
    print(path.split('/')[-2], '  ', prompt)

    d = target - source
    d_norm_max = torch.norm(d.float(), dim=1, keepdim=True).max()

    if d_norm_max <= opt.step_size:
        drag_num = 1
    else:
        drag_num = d_norm_max.div(torch.tensor([opt.step_size]), rounding_mode='trunc')
        if (d_norm_max / drag_num - opt.step_size).abs() > (d_norm_max / (drag_num + 1) - opt.step_size).abs():
            drag_num += 1

    guidance_scale = opt.scale

    with torch.autocast(torch_device):
        img = get_img_latent(path, vae, 512, 512)
    train_lora_and_load(path, prompt, unet)
    mask = mask.unsqueeze(0).expand_as(img).to(torch_device)

    with torch.autocast(torch_device):
        text_embeddings = get_text_embed(prompt, tokenizer, text_encoder)
        uncond_embeddings = get_text_embed([""], tokenizer, text_encoder)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    imgs = []
    with torch.autocast(torch_device):
        for i in range(int(drag_num)):
            r = opt.r
            source_new = source + (i / drag_num * d).to(torch.int)
            target_new = source + ((i + 1) / drag_num * d).to(torch.int)

            latents, noises, hook_latents, lora_scales, cfg_scales = forward(opt, img)
            latents = copy_and_paste(latents, source_new, target_new)
            img = backward(opt, latents, noises, hook_latents, lora_scales, cfg_scales)
            imgs.append(img)

    # scale and decode the image latents with vae
    imgs = 1 / 0.18215 * imgs[-1]
    with torch.autocast(torch_device):
        with torch.no_grad():
            imgs = vae.decode(imgs).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    save_path = f'output/sdedrag_dragbench'
    os.makedirs(save_path, exist_ok=True)

    file_name = path.split('/')[-2]
    save_image(imgs, os.path.join(save_path, f'{file_name}.png'))


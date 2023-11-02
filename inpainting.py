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


import os
import argparse

import torch

from diffusers import (DPMSolverMultistepScheduler,
                       StableDiffusionInpaintPipeline)
from PIL import Image
from cycle_sde import set_seed


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
            default='assets/inpainting',
            help="origin image and mask path"
    )
    parser.add_argument(
            "--steps",
            type=int,
            default=50,
            help="sampling steps"
    )
    parser.add_argument(
            "--sde",
            action='store_true',
            help="use inpainting-sde",
    )
    parser.add_argument(
            "--order",
            type=int,
            default=1,
            help='solver order'
    )

    opt = parser.parse_args()
    return opt


def main():
    opt = get_args()
    set_seed(opt.seed)

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline.enable_model_cpu_offload()

    if opt.sde:
        algorithm = 'sde-dpmsolver++'
    else:
        algorithm = 'dpmsolver++'
    num_inference_steps = opt.steps

    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    scheduler.config.algorithm_type = algorithm
    scheduler.config.solver_order = opt.order
    pipeline.scheduler = scheduler

    init_image = Image.open(os.path.join(opt.img_path, 'origin.png')).resize((512, 512))
    mask_image = Image.open(os.path.join(opt.img_path, 'mask.png')).resize((512, 512))

    prompt = "Face of a cat, high resolution, sitting on a park bench"
    image, _ = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, return_dict=False,
                        num_inference_steps=num_inference_steps)

    path = 'output/inpainting'
    os.makedirs(path, exist_ok=True)
    image[0].save(os.path.join(path, f'{algorithm}-order={opt.order}.png'))


if __name__ == "__main__":
    main()
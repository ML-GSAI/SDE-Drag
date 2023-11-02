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

import json
import os
from copy import deepcopy

import cv2
import gradio as gr
import numpy as np
import PIL.Image
from PIL import Image

from .drag import drag
from .train_lora import train_lora


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]  # center crop
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]

    return np.array(img).astype(np.uint8)

# user click the image to get points, and show the points on the image
def get_point(img, sel_pix, evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

def mask_image(image, mask, color=[255,0,0], alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out


def run_process(original_image, input_image, mask, selected_points, prompt_textbox, output_path,
                model_path, save_lora_path, lora_step,
                drag_t, steps, step_size, image_scale, adapt_r, use_lora, lora_scale_min, seed):
    """ When the "run" button is pressed, the function goes through the following process:
            1. It calls the save_data function to save the user's input image, mask, prompt, source point, 
                and target point to the directory './drag_data/`output_path`'.
            2. If the use_lora parameter is set to True, it calls the train_lora function to train the input
                image and save the corresponding files to the directory './`save_lora_path`/`output_path`'.
            3. For the convenience of functions get_img_latent to read image, it reads the information saved by the
                save_data function and performs dragging based on the corresponding image address, mask address, prompt,
                and relevant settings. It returns the drag's state and the resulting image. 
        The results obtained during each drag are saved in './output/`output_path`'.
    """    
    save_data(original_image, input_image, mask, selected_points, prompt_textbox, output_path)
    if use_lora:
        train_lora(original_image, prompt_textbox, model_path, save_lora_path, output_path, lora_step, progress=gr.Progress())

    path = os.path.join('drag_data', output_path)
    image_path = os.path.join(path, 'origin_image.png')
    mask_path = os.path.join(path, 'mask.png')
    with open(os.path.join(path, 'prompt.json'), 'r') as f:
        prompt = json.load(f)
        
    save_lora_path = os.path.join(save_lora_path, output_path)
    os.makedirs(os.path.join("output", output_path), exist_ok=True)
    os.makedirs(save_lora_path, exist_ok=True)

    for state_text, img in drag(image_path, mask_path, prompt['prompt'], prompt['source'], prompt['target'], drag_t, steps, step_size, image_scale, adapt_r, use_lora, lora_scale_min, save_lora_path, output_path, seed):
        yield state_text, img
    

def save_data(original_image, input_image, mask, selected_points, prompt, output_path):
    """Save the user input image, its corresponding mask, prompts, as well as the source and target 
        points to the directory path './grad_data/output_path'.
    """      
    path = os.path.join('drag_data', output_path)
    os.makedirs(path, exist_ok=True)

    original_image = PIL.Image.fromarray(original_image)
    original_image.save(os.path.join(path, 'origin_image.png'))

    input_image = PIL.Image.fromarray(input_image)
    input_image.save(os.path.join(path, 'input_image.png'))

    mask = mask * 255
    if mask.sum() == 0:
        mask = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask, 'L')
    mask.save(os.path.join(path, 'mask.png'))

    assert len(selected_points) % 2 == 0
    sourece_point = selected_points[::2]
    target_point = selected_points[1::2]

    with open(os.path.join(path, 'prompt.json'), "w") as f:
        json.dump({'source': sourece_point, 'target': target_point, 'prompt': prompt}, f)


# Once user upload an image, the original image is stored in `original_image`,
# the same image is displayed in `input_image` for point clicking purpose
def store_img(img):    
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # When new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


# Clear all handle/target points
def undo_points(original_image, mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []

def upload_point_image(input_image):
    return "Upload Failed. Please upload image via the leftmost canvas", None

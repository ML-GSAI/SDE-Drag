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

import gradio as gr

from utils.ui_utils import (get_point, store_img, undo_points, run_process, upload_point_image)

with gr.Blocks() as demo:
    with gr.Tab(label="Image"):
        with gr.Row():
            gr.Markdown("""
                        Operating Instructions:

                        **Step 1.** On the leftmost canvas, upload the image you wish to edit and draw the editing region (mask).

                        **Step 2.** On the middle canvas, click to select the starting and destination regions you wish to edit. Please note that the starting and destination regions must correspond one-to-one. If you need to make a new selection, click the 'Undo point' button.

                        **Step 3.** Choose whether to use LoRA and enter the corresponding LoRA prompt in the text box.

                        **Step 4.** Specify the output path for the image, which is set to the default location './output/default' within the project directory.

                        **Step 5.** Configure the relevant hyperparameters at the bottom (if you are unsure about the functions of specific hyperparameters, it is advisable to refer to the hyperparameter documentation or keep the default settings).

                        **Step 6.** Click the 'run' button and wait until the editing is complete.

                        The 'State' textbox will display the processing status or error messages related to the operation.
                        """)

        with gr.Row():
            original_image = gr.State(value=None)
            mask = gr.State(value=None)
            selected_points = gr.State([])

            length = 400
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Draw Mask</p>""")
                canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask", show_label=True,
                                  show_download_button=True, height=length, width=length)
                with gr.Row():
                    use_lora = gr.Checkbox(label="Use LoRA")

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points", show_label=True, show_download_button=True,
                                       height=length, width=length)
                with gr.Row():
                    undo_button = gr.Button("Undo point")

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Editing results</p>""")
                editing_result = gr.Image(type="numpy", label="Editing Results", show_label=True,
                                          show_download_button=True, height=length, width=length)
                with gr.Row():
                    run_button = gr.Button("Run")

        with gr.Row():
            with gr.Column():
                state_textbox = gr.Textbox(label="State")

        with gr.Row():
            with gr.Column():
                prompt_textbox = gr.Textbox(label="Prompt")

        with gr.Row():
            output_path = gr.Textbox(value='default', label="Output path(eg: bear)")

        with gr.Row():
            with gr.Tab("LoRA Parameters"):
                with gr.Row():
                    model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                                             label="Diffusion model",
                                             choices=["runwayml/stable-diffusion-v1-5"]
                                             )
                    lora_path_textbox = gr.Textbox(label="LoRA path", value='./lora')
                    lora_step = gr.Number(value=100, label="LoRA training steps", precision=0)
                    lora_scale_min = gr.Number(value=0.5, label="Min LoRA scale")

        with gr.Row():
            with gr.Tab("Drag Parameters"):
                with gr.Row():
                    drag_t = gr.Number(value=0.6, label="t0")
                    steps = gr.Number(value=100, label="Sampling steps", precision=0)
                    step_size = gr.Number(value=2, label="Step size")
                    img_scale = gr.Number(value=0.3, label="beta")
                    adapt_r = gr.Number(value=5, label="r", precision=0)
                    seed = gr.Number(value=1234, label="seed", precision=0)

        with gr.Row():
            gr.Markdown("""
                        ### Parameters in LoRA Parameters:

                        Diffusion model: The diffusion model used, with a default setting of 'runwayml/stable-diffusion-v1.5.'
                        
                        LoRA path: When 'Use LoRA' is enabled, this parameter specifies the location to save the model file resulting from training LoRA on the input image. The default path is './lora/(Output path from the text box).
                        
                        LoRA training steps: The number of steps for LoRA fine-tuning.
                        
                        Min LoRA scale: We reduce the LoRA scale from 1 to (Min LoRA scale) as time goes from 0 to t0. When this value is set to $1.0$, the LoRA scale remains constant.

                        ### Parameters in Drag Parameters:

                        t0: The time parameter $t_0$. Higher $t_0$ improves the fidelity while lowering the faithfulness of the edited images and vice versa. Set $t_0 = 0.6T$ by default.
                        
                        Sampling Steps: The number of sampling iterations used in the DDPM.
                        
                        Step Size: The drag diatance of each drag process. Increasing this parameter may improve the speed of image editing but could potentially impact the quality of results.
                        
                        beta: Beta is used to control the attenuation of noise in each copy and paste operation to prevent image redundancy.
                        
                        r: The size of the region for copy and paste operations during each step of the drag process.
                        
                        seed: Random seed. SDE-Drag is a stochastic algorithm where different random seeds may yield different outcomes. .
                        """)

    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )

    input_image.upload(
        upload_point_image,
        [input_image],
        [state_textbox, input_image]
    )

    input_image.select(
        get_point,
        [input_image, selected_points],
        [input_image],
    )

    run_button.click(
        run_process,
        [original_image, input_image, mask, selected_points, prompt_textbox, output_path,
         model_path, lora_path_textbox, lora_step,
         drag_t, steps, step_size, img_scale, adapt_r, use_lora, lora_scale_min, seed],
        [state_textbox, editing_result]
    )

    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )

demo.queue().launch(share=True, debug=True)

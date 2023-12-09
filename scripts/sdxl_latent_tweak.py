# Copyright 2023 SLAPaper
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

import sys

import gradio as gr
import torch

import modules.processing as mp
import modules.script_callbacks as msc
import modules.scripts as ms


class SdxlLatentTweaking(ms.Script):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        return "SDXL Latent Fixing"

    def show(self, is_img2img: bool):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
        """
        return ms.AlwaysVisible

    def ui(self, is_img2img: bool):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                with gr.Column():
                    enable_clamping = gr.Checkbox(label="Enable Clamping", value=False)
                    clamping_factor = gr.Slider(
                        label="Clamping Factor",
                        value=0.998,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                    )
                clamping_start = gr.Slider(
                    label="Clamping Start",
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                clamping_end = gr.Slider(
                    label="Clamping End",
                    value=0.05,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
            with gr.Row():
                with gr.Column():
                    enable_centering = gr.Checkbox(
                        label="Enable Centering", value=False
                    )
                    centering_channels = gr.CheckboxGroup(
                        label="Centering Channels",
                        choices=[("Brightness", "0"), ("Color", "1,2"), ("Other", "3")],
                    )
                centering_start = gr.Slider(
                    label="Centering Start",
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                centering_end = gr.Slider(
                    label="Centering End",
                    value=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
            with gr.Row():
                with gr.Column():
                    enable_maximizing = gr.Checkbox(
                        label="Enable Maximizing", value=False
                    )
                    maximizing_channels = gr.CheckboxGroup(
                        label="Maximizing Channels",
                        choices=[("Brightness", "0"), ("Color", "1,2"), ("Other", "3")],
                    )
                maximizing_start = gr.Slider(
                    label="Maximizing Start",
                    value=0.9,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                maximizing_end = gr.Slider(
                    label="Maximizing End",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
            enable_debug_log = gr.Checkbox(label="Enable Debug Log", value=False)

        return [
            enable_clamping,
            clamping_factor,
            clamping_start,
            clamping_end,
            enable_centering,
            centering_channels,
            centering_start,
            centering_end,
            enable_maximizing,
            maximizing_channels,
            maximizing_start,
            maximizing_end,
            enable_debug_log,
        ]

    def process(self, p: mp.StableDiffusionProcessing, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        msc.remove_current_script_callbacks()

        (
            enable_clamping,
            clamping_factor,
            clamping_start,
            clamping_end,
            enable_centering,
            centering_channels,
            centering_start,
            centering_end,
            enable_maximizing,
            maximizing_channels,
            maximizing_start,
            maximizing_end,
            enable_debug_log,
        ) = args

        def denoise_callback(params: msc.CFGDenoisedParams):
            """callback of denoise process"""

            current_step = params.sampling_step
            total_step = params.total_sampling_steps

            def print_debug_log(stage: str) -> None:
                """print debug log"""
                if not enable_debug_log:
                    return

                print(
                    f"SDXL Latent Fixing DEBUG: {stage}",
                    f"({current_step}/{total_step})",
                    f"(size:{params.x.size()})",
                    f"(chmax:{torch.amax(params.x, (0, 2, 3))})",
                    f"(chmin:{torch.amin(params.x, (0, 2, 3))})",
                    f"(mean:{torch.mean(params.x)})",
                    "...",
                    file=sys.stderr,
                )

            if (
                enable_clamping
                and current_step >= clamping_start * total_step
                and current_step <= clamping_end * total_step
            ):
                upper = torch.abs(torch.max(params.x))
                lower = torch.abs(torch.min(params.x))
                print_debug_log("before soft clamping")
                threshold = torch.max(upper, lower) * clamping_factor
                params.x = soft_clamp_tensor(
                    params.x,
                    threshold=threshold * clamping_factor,
                    boundary=threshold,
                )
                print_debug_log("after soft clamping")

            if (
                enable_centering
                and current_step >= centering_start * total_step
                and current_step <= centering_end * total_step
            ):
                print_debug_log("before centering")
                channels = [int(x) for x in ",".join(centering_channels).split(",")]
                params.x = center_tensor(
                    params.x,
                    0.8,
                    0.8,
                    channels=channels,
                )
                print_debug_log("after centering")

            if (
                enable_maximizing
                and current_step >= maximizing_start * total_step
                and current_step <= maximizing_end * total_step
            ):
                print_debug_log("before maximizing")
                channels = [int(x) for x in ",".join(maximizing_channels).split(",")]
                params.x = center_tensor(
                    params.x,
                    0.6,
                    1.0,
                    channels=channels,
                )
                print_debug_log("in maximizing")
                params.x = maximize_tensor(
                    params.x,
                    channels=channels,
                )
                print_debug_log("after maximizing")

        msc.on_cfg_denoised(denoise_callback)


# Shrinking towards the mean (will also remove outliers)
def soft_clamp_tensor(
    input_tensor: torch.Tensor,
    threshold: float | torch.Tensor = 3.5,
    boundary: float | torch.Tensor = 4,
):
    """huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#outlier-removal"""
    if (
        torch.max(torch.abs(input_tensor.max()), torch.abs(input_tensor.min()))
        < threshold
    ):
        return input_tensor

    channel_dim = 1

    max_vals = input_tensor.max(channel_dim, keepdim=True)[0]
    max_replace = ((input_tensor - threshold) / (max_vals - threshold)) * (
        boundary - threshold
    ) + threshold
    over_mask = input_tensor > threshold

    min_vals = input_tensor.min(channel_dim, keepdim=True)[0]
    min_replace = ((input_tensor + threshold) / (min_vals + threshold)) * (
        -boundary + threshold
    ) - threshold
    under_mask = input_tensor < -threshold

    return torch.where(
        over_mask, max_replace, torch.where(under_mask, min_replace, input_tensor)
    )


# Center tensor (balance colors)
def center_tensor(
    input_tensor: torch.Tensor,
    per_channel_shift: float = 1.0,
    full_tensor_shift: float = 1.0,
    channels=[1, 2],
):
    """huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#color-balancing-and-increased-range"""
    for channel in channels:
        input_tensor[:, channel] -= input_tensor[:, channel].mean() * per_channel_shift
    return input_tensor - input_tensor.mean() * full_tensor_shift


# Maximize/normalize tensor
def maximize_tensor(
    input_tensor: torch.Tensor,
    boundary: float = 4.0,
    channels=[0, 1, 2],
):
    """huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#color-balancing-and-increased-range"""
    min_val = input_tensor.min()
    max_val = input_tensor.max()

    normalization_factor = boundary / torch.max(torch.abs(min_val), torch.abs(max_val))
    input_tensor[:, channels] *= normalization_factor

    return input_tensor

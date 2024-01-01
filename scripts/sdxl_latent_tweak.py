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

import functools as ft
import itertools as it
import sys
import typing as tg

import gradio as gr
import gradio.components.base as grcb
import torch

import modules.processing as mp
import modules.script_callbacks as msc
import modules.scripts as ms

_G_CURR_STATE: dict[str, tg.Any] = {}


class SdxlLatentTweaking(ms.Script):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        return "SDXL Latent Tweaking"

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
                        choices=["Brightness", "Color", "Pattern"],
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
                        choices=["Brightness", "Color", "Pattern"],
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

        self.infotext_fields = [  # type: ignore
            (enable_clamping, "Latent Soft Clamping"),
            (clamping_factor, "Latent Soft Clamping Factor"),
            (
                clamping_start,
                lambda d: d["Latent Soft Clamping Range"].split(";")[0]
                if "Latent Soft Clamping Range" in d
                and ";" in d["Latent Soft Clamping Range"]
                else gr.update(),
            ),
            (
                clamping_end,
                lambda d: d["Latent Soft Clamping Range"].split(";")[1]
                if "Latent Soft Clamping Range" in d
                and ";" in d["Latent Soft Clamping Range"]
                else gr.update(),
            ),
            (enable_centering, "Latent Centering"),
            (
                centering_channels,
                lambda d: d["Latent Centering Channels"].split(";")
                if "Latent Centering Channels" in d
                and ";" in d["Latent Centering Channels"]
                else gr.update(),
            ),
            (
                centering_start,
                lambda d: d["Latent Centering Range"].split(";")[0]
                if "Latent Centering Range" in d and ";" in d["Latent Centering Range"]
                else gr.update(),
            ),
            (
                centering_end,
                lambda d: d["Latent Centering Range"].split(";")[1]
                if "Latent Centering Range" in d and ";" in d["Latent Centering Range"]
                else gr.update(),
            ),
            (enable_maximizing, "Latent Maximizing"),
            (
                maximizing_channels,
                lambda d: d["Latent Maximizing Channels"].split(";")
                if "Latent Maximizing Channels" in d
                and ";" in d["Latent Maximizing Channels"]
                else gr.update(),
            ),
            (
                maximizing_start,
                lambda d: d["Latent Maximizing Range"].split(";")[0]
                if "Latent Maximizing Range" in d
                and ";" in d["Latent Maximizing Range"]
                else gr.update(),
            ),
            (
                maximizing_end,
                lambda d: d["Latent Maximizing Range"].split(";")[1]
                if "Latent Maximizing Range" in d
                and ";" in d["Latent Maximizing Range"]
                else gr.update(),
            ),
        ]

        self.paste_field_names = [f for _, f in self.infotext_fields]  # type: ignore

        msc.on_cfg_denoised(SdxlLatentTweaking.denoise_callback)

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

        _G_CURR_STATE["enable_clamping"] = enable_clamping
        _G_CURR_STATE["clamping_factor"] = clamping_factor
        _G_CURR_STATE["clamping_start"] = clamping_start
        _G_CURR_STATE["clamping_end"] = clamping_end
        _G_CURR_STATE["enable_centering"] = enable_centering
        _G_CURR_STATE["centering_channels"] = centering_channels
        _G_CURR_STATE["centering_start"] = centering_start
        _G_CURR_STATE["centering_end"] = centering_end
        _G_CURR_STATE["enable_maximizing"] = enable_maximizing
        _G_CURR_STATE["maximizing_channels"] = maximizing_channels
        _G_CURR_STATE["maximizing_start"] = maximizing_start
        _G_CURR_STATE["maximizing_end"] = maximizing_end
        _G_CURR_STATE["enable_debug_log"] = enable_debug_log

        if enable_clamping:
            p.extra_generation_params["Latent Soft Clamping"] = True
            p.extra_generation_params["Latent Soft Clamping Factor"] = clamping_factor
            p.extra_generation_params[
                "Latent Soft Clamping Range"
            ] = f"{clamping_start};{clamping_end}"

        if enable_centering:
            p.extra_generation_params["Latent Centering"] = True
            p.extra_generation_params["Latent Centering Channels"] = ";".join(
                centering_channels
            )
            p.extra_generation_params[
                "Latent Centering Range"
            ] = f"{centering_start};{centering_end}"

        if enable_maximizing:
            p.extra_generation_params["Latent Maximizing"] = True
            p.extra_generation_params["Latent Maximizing Channels"] = ";".join(
                maximizing_channels
            )
            p.extra_generation_params[
                "Latent Maximizing Range"
            ] = f"{maximizing_start};{maximizing_end}"

    @staticmethod
    def denoise_callback(params: msc.CFGDenoisedParams):
        """callback of denoise process"""

        current_step = params.sampling_step
        total_step = params.total_sampling_steps

        def print_debug_log(stage: str) -> None:
            """print debug log"""
            if not _G_CURR_STATE["enable_debug_log"]:
                return

            print(
                f"SDXL Latent Tweaking DEBUG: {stage}",
                f"({current_step}/{total_step})",
                f"(size:{params.x.size()})",
                f"(chmax:{torch.amax(params.x, (0, 1, 2, 3))})",
                f"(chmin:{torch.amin(params.x, (0, 1, 2, 3))})",
                f"(mean:{torch.mean(params.x)})",
                "...",
                file=sys.stderr,
            )

        if (
            _G_CURR_STATE["enable_clamping"]
            and current_step >= _G_CURR_STATE["clamping_start"] * total_step
            and current_step <= _G_CURR_STATE["clamping_end"] * total_step
        ):
            upper = torch.abs(torch.max(params.x))
            lower = torch.abs(torch.min(params.x))
            print_debug_log("before soft clamping")
            threshold = torch.max(upper, lower) * _G_CURR_STATE["clamping_factor"]
            params.x = soft_clamp_tensor(
                params.x,
                threshold=threshold * _G_CURR_STATE["clamping_factor"],
                boundary=threshold,
            )
            print_debug_log("after soft clamping")

        if (
            _G_CURR_STATE["enable_centering"]
            and current_step >= _G_CURR_STATE["centering_start"] * total_step
            and current_step <= _G_CURR_STATE["centering_end"] * total_step
        ):
            print_debug_log("before centering")
            params.x = center_tensor(
                params.x,
                0.8,
                0.8,
                channels=channel_name_to_channel_index(
                    _G_CURR_STATE["centering_channels"]
                ),
            )
            print_debug_log("after centering")

        if (
            _G_CURR_STATE["enable_maximizing"]
            and current_step >= _G_CURR_STATE["maximizing_start"] * total_step
            and current_step <= _G_CURR_STATE["maximizing_end"] * total_step
        ):
            print_debug_log("before maximizing")
            # use v2 maximizing for brightness channel
            selected_channel_names: list[str] = _G_CURR_STATE["maximizing_channels"]
            if "Brightness" in selected_channel_names:
                selected_channel_names.remove("Brightness")
                params.x = maximize_tensor_v2(
                    params.x, channels=channel_name_to_channel_index(["Brightness"])
                )
                print_debug_log("after brightness v2 maximizing")

            channels = channel_name_to_channel_index(selected_channel_names)
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


CHANNEL_NAME_TO_INDEX: dict[str, list[int]] = {
    "Brightness": [0],
    "Color": [1, 2],
    "Pattern": [3],
}


def channel_name_to_channel_index(name_list: list[str]) -> list[int]:
    """convert channel name to index"""
    return list(
        it.chain.from_iterable(
            CHANNEL_NAME_TO_INDEX[name]
            for name in name_list
            if name in CHANNEL_NAME_TO_INDEX
        )
    )


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


# Maximize tensor (v2)
def maximize_tensor_v2(
    input_tensor: torch.Tensor,
    boundary: float = 4.0,
    channels=[0, 1, 2],
):
    """like the 'level' in PhotoShop, scale min/max value while keeping the mean

    Input: (batch, channel, width, height)
    """
    for i in range(input_tensor.size(0)):
        # Select the specific channel for this batch item
        channel_data = input_tensor[i, channels, :, :]

        # Check if data contains both positive and negative values
        if torch.any(channel_data < 0) and torch.any(channel_data > 0):
            # Calculate the mean
            mean = channel_data.mean()

            # Scale values differently based on their relation to the mean
            channel_data = torch.where(
                channel_data < mean,
                -boundary * (channel_data / channel_data[channel_data < mean].min()),
                channel_data,
            )
            channel_data = torch.where(
                channel_data > mean,
                boundary * (channel_data / channel_data[channel_data > mean].max()),
                channel_data,
            )
        else:
            # Rescale such that the absolute maximum value is boundary
            max_abs_val = torch.max(torch.abs(channel_data))
            if max_abs_val > 0:
                channel_data = boundary * channel_data / max_abs_val

        # Update the tensor
        input_tensor[i, channels, :, :] = channel_data

    return input_tensor

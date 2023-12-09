# sd-webui-sdxl-latent-tweaking

Implementing 3 tweaks introduced in [HF Blog](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space), go check the blog out if you are interested in the details.

## TL;DR

SDXL's latent has 4 channels: Brightness, Color1(Cyan/Red), Color2(Lime/Medium Purple), Other(Pattern/Structure)

by tweaking those 4 channels, you can get a lot of interesting results. For more manual control, checkout [CD Tuner](https://github.com/hako-mikan/sd-webui-cd-tuner)

In this extension, the following 3 tweaks are implemented:

- Soft Clamping: by remove outliers in the latent space, less messy objects will be generated. This is intended to be activated during begining of sampling.
- Centering Latent: by centering the latent space, you will get more "neutral" results. Depending by the selected channel, you can get "auto exposure" effect (Contrast Channel), "White ballance" effect (Color Channel), "Minor improvement" effect (Other Channel), or all above.
- Maximizing Latent: by maximizing the latent space (-4.0~4.0), you will get some "HDR" looking image. The effect is also dependent on the selected channel.

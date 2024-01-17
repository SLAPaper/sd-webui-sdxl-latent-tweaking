# sd-webui-sdxl-latent-tweaking

Implementing 3 tweaks introduced in [HF Blog](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space), go check the blog out if you are interested in the details.

## TL;DR

SDXL's latent has 4 channels, first three are CIELAB color space, the last is pattern/structure: Brightness(L*), Color1(negative a*), Color2(b*), Pattern/Structure

by tweaking those 4 channels, you can get a lot of interesting results. For more manual control, checkout [CD Tuner](https://github.com/hako-mikan/sd-webui-cd-tuner)

In this extension, the following 3 tweaks are implemented:

- Soft Clamping: by remove outliers in the latent space, less messy objects will be generated. This is intended to be activated during begining of sampling.
- Centering Latent: by centering the latent space, you will get more "neutral" results. Depending by the selected channel, you can get "auto exposure" effect (Contrast Channel), "White ballance" effect (Color Channel), "Minor improvement" effect (Other Channel), or all above.
- Maximizing Latent: by maximizing the latent space (-4.0~4.0), you will get some "HDR" looking image. The effect is also dependent on the selected channel. In latest version, the Brightness channel has specific operation to mimic 'level' in PhotoShop, so that the half tone won't shift away.

## For SD1.5

According to [sd-webui-diffusion-cg](https://github.com/Haoming02/sd-webui-diffusion-cg?tab=readme-ov-file#stable-diffusion-structures), SD1.5's 4 channels are: Negative Black, Negative Magenta, Cyan, Yellow, so if you want to tweak color, you have to select both `Color` and `Pattern` to get correct results.

## New option for Hires Fix

If you find hires fix shifting the color off, you may disable the tweakings when hires fix is activated by checking `Disable When HR Fix`.

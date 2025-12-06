import numpy as np
from PIL import Image
import torch
import wandb
import tempfile
import os
from typing import List, Optional, Tuple, Union
import torch
from diffusers import UNet2DModel, DiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.autograd import Variable


def predict_x0_from_xt(x_t, t, noise_pred, scheduler):
    # Get alpha_bar_t = scheduler.alphas_cumprod[t]
    if isinstance(t, int):
        t = torch.tensor([t], device=x_t.device)

    # Get alpha_bar_t from scheduler
    alpha_cumprod = scheduler.alphas_cumprod.to(x_t.device)  # [num_timesteps]
    alpha_bar_t = alpha_cumprod[t].reshape(-1, 1, 1, 1)       # [B, 1, 1, 1]

    # Compute x0
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    return x0_pred


def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
                
    @torch.no_grad()
    def __call__(
        self,
        conditioning,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        batch_size = conditioning.shape[0]
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states=conditioning).sample
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()
        
        image = image.permute(0, 2, 3, 1)
        
        return image 

def log_images_to_wandb(accelerator, images, step):
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, image in enumerate(images):
            pil = Image.fromarray(
                (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            
            pil = pil.resize((256, 256))
            pil.save(os.path.join(tmpdir, f"{i}.jpg"))

        if accelerator.is_main_process:
            accelerator.log(
            {
                "images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{i}.jpg"),
                        caption=f"i",
                    )
                    for i in (
                        range(len(images))
                    )  # only log rewards from process 0
                ],
            },
            step=step,
        )
            
            
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid
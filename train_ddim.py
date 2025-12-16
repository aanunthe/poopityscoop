import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['WANDB_MODE'] = 'online'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand for better stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms
from utils import log_images_to_wandb, grid_sample, predict_x0_from_xt
import datetime
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
import torch
import subprocess
import sys
import glob
import shutil
from torch.optim import AdamW
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from parse_args import parse_args
import torch
import torch.nn.functional as F
from data_loader import Doc3d
from transformers import CLIPProcessor, CLIPModel
from typing import List, Optional, Tuple, Union
from diffusers import UNet2DModel, DiffusionPipeline
from diffusers.pipelines import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import DDIMScheduler

class FastDDIMPipeline(DiffusionPipeline):
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

    def __init__(self, unet: UNet2DModel, scheduler: DDIMScheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 20,  # Reduced from 50 for faster inference
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

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

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Use torch.compile for faster inference (PyTorch 2.0+)
        # if hasattr(torch, 'compile'):
        #     compiled_unet = torch.compile(self.unet, mode='reduce-overhead')
        # else:
        compiled_unet = self.unet

        # Batch processing for faster inference
        with torch.amp.autocast('cuda'):
            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = compiled_unet(image, t, encoder_hidden_states=prompt_embeds).sample
                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                image = self.scheduler.step(
                    model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                ).prev_sample


        # Handle different output types
        if output_type == "pil":
            # Normalize from [-1, 1] to [0, 1] for proper PIL conversion
            image = (image + 1.0) / 2.0
            image = torch.clamp(image, 0.0, 1.0)
            image = image.permute(0, 2, 3, 1)
        elif output_type == "pt":
            # Keep raw tensor format for PyTorch operations
            # Don't permute dimensions for pt output
            pass
        else:
            # Default behavior - permute for numpy/other formats
            image = image.permute(0, 2, 3, 1)
        
        return image



def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def main(args):
    # Setup for 4-GPU training
    if args.gpu_ids == "0,1,2,3":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        # Enable optimizations for multi-GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = f"ddim_fast_{unique_id}"
    else:
        args.run_name = f"ddim_fast_{args.run_name}_{unique_id}"

    # 🛠️ Change 1: Update ProjectConfiguration
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(args.output_dir, args.run_name),
        automatic_checkpoint_naming=True, # Enable automatic naming for limit to work
        total_limit=2 # Keep only the 2 most recent checkpoints
    )
    
    # Enhanced accelerator for multi-GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_config,
        log_with='wandb',
        mixed_precision='fp16',
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={
                "wandb": {"name": args.run_name}
            }
        )
    
    device = accelerator.device
    set_seed(args.seed)
    
    # Load dataset with optimized data loading
    dataset = Doc3d(args.dataset_name, args, resize=True)
    
    # Optimized DataLoader for multi-GPU training
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(8, mp.cpu_count()),  # Optimal number of workers
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
        drop_last=True,  # Ensure consistent batch sizes across GPUs
    )

    # Optimized UNet model for faster training
    model = UNet2DConditionModel(
        sample_size=288,
        in_channels=2,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),  # Reduced channels for speed
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
        encoder_hid_dim=768,
        attention_head_dim=64,  # Optimized attention heads
        use_linear_projection=True,  # More efficient attention
        only_cross_attention=False,
        upcast_attention=False,  # Keep attention in fp16
    )
    
    # Load CLIP model for conditioning with optimizations
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=True)
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    
    # Freeze CLIP model for faster training
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
    
    # Move to device after accelerator preparation
    clip_model = clip_model.to(device)

    # Initialize optimizer with optimizations for multi-GPU
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=True,  # Faster optimizer step
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    # Use DDIM scheduler with optimized settings
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Compile model for faster training (PyTorch 2.0+)
    # if hasattr(torch, 'compile') and accelerator.is_main_process:
    #     print("Compiling model for faster training...")
    #     model = torch.compile(model, mode='reduce-overhead')
    
    # Training Loop with optimizations
    global_step = 0
    
    for epoch in tqdm(range(args.num_epochs), leave=False, disable=not accelerator.is_main_process, desc='Epochs'):
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader, leave=False, disable=not accelerator.is_main_process, desc='Data')):
            # Move data to device efficiently
            img = batch['img']
            latent = batch['bm']

            #STEP CAP
            if step >= 150:
                break
            
            # 💡 FIX: Normalize the image tensor to the [0, 1] range before passing to the CLIP processor
            img_normalized = (img + 1.0) / 2.0
            
            # Pre-compute CLIP features with caching
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # Process images in batch for efficiency
                clip_inputs = processor(images=img_normalized, return_tensors='pt', do_rescale=False)
                clip_inputs = {k: v.to(device, non_blocking=True) for k, v in clip_inputs.items()}
                encoder_hidden_states = clip_model.get_image_features(**clip_inputs)  # [B, 768]
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)          # [B, 1, 768]

            # Efficient noise sampling
            noise = torch.randn_like(latent, device=device, dtype=latent.dtype)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latent.shape[0],), device=device, dtype=torch.long
            )
            noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    # Forward pass
                    noise_pred = model(
                        noisy_latent,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample

                    # Optimized loss computation
                    loss_elbo = F.mse_loss(noise_pred, noise, reduction='mean')

                    # Efficient x0 reconstruction
                    x0_pred = predict_x0_from_xt(noisy_latent, timesteps, noise_pred, noise_scheduler)
                    loss_recon = F.mse_loss(x0_pred, latent, reduction='mean')

                    loss = loss_elbo + args.alpha * loss_recon

                # Optimized backward pass
                accelerator.backward(loss)
                
                # Gradient clipping with reduced norm
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                
                torch.cuda.empty_cache()

            if accelerator.is_main_process and global_step % args.log_freq == 0:
                accelerator.log({
                    "loss": loss.item(),
                    "loss_elbo": loss_elbo.item(),
                    "loss_recon": loss_recon.item(),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr']
                })

            # 🛠️ Change 2: Modify checkpointing logic
            if accelerator.is_main_process and global_step % args.save_freq == 0:
                # Manual Checkpoint Rotation
                # Get list of existing checkpoints
                save_dir = accelerator_config.project_dir
                if os.path.exists(save_dir):
                    checkpoints = sorted([d for d in os.listdir(save_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1]))
                    
                    # Keep only the most recent 1 checkpoint to save massive space
                    if len(checkpoints) >= 1:
                        for d in checkpoints[:-1]: # Delete all except the absolute latest
                             path_to_remove = os.path.join(save_dir, d)
                             print(f"Removing old checkpoint to save space: {path_to_remove}")
                             shutil.rmtree(path_to_remove, ignore_errors=True)

                # Save model weights only (lighter than full state) and compatible with eval script
                checkpoint_dir = os.path.join(save_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
                
                pipeline = FastDDIMPipeline(accelerator.unwrap_model(model), noise_scheduler)
                evaluate(args, epoch, encoder_hidden_states, img, pipeline, accelerator, global_step)
        accelerator.wait_for_everyone()

    # --- End of Training ---
    # Save final model
    if accelerator.is_main_process:
        final_checkpoint_dir = os.path.join(accelerator_config.project_dir, f"checkpoint-{global_step}")
        # Save only the UNet for evaluation (more compatible format)
        unwrap_model = accelerator.unwrap_model(model)
        unwrap_model.save_pretrained(final_checkpoint_dir)
        print(f"Saved final checkpoint to {final_checkpoint_dir}")
        
        # Free up memory before evaluation
        del model, optimizer, train_dataloader, lr_scheduler
        torch.cuda.empty_cache()

        if args.do_eval:
            print("Starting full evaluation pipeline...")
            
            # 1. Run Generation
            eval_output_dir = os.path.join(accelerator_config.project_dir, "eval_results")
            
            # Use only the first GPU for evaluation scripts to avoid complexity
            # Extract the first ID from the comma-separated list
            first_gpu = args.gpu_ids.split(',')[0]
            
            gen_cmd = [
                sys.executable, "generate_eval_images.py",
                "--checkpoint_path", final_checkpoint_dir,
                "--dataset_name", args.dataset_name,
                "--output_dir", eval_output_dir,
                "--image_size", str(args.image_size),
                "--batch_size", str(args.batch_size),
                "--gpu_ids", first_gpu
            ]
            
            print(f"Running generation: {' '.join(gen_cmd)}")
            subprocess.run(gen_cmd, check=True)
            
            # 2. Run Metrics
            metrics_cmd = [
                sys.executable, "evaluation_metrics.py",
                "--rec_path", os.path.join(eval_output_dir, "rectified"),
                "--gt_path", os.path.join(eval_output_dir, "ground_truth")
            ]
            
            print(f"Running metrics: {' '.join(metrics_cmd)}")
            subprocess.run(metrics_cmd, check=True)
            
            print("Evaluation completed successfully.")



@torch.no_grad()
def evaluate(args, epoch, encoder_hidden_states, img, pipeline, accelerator, global_step):
    pipeline.unet.eval()
    
    # Faster evaluation with fewer steps
    bm = pipeline(
        batch_size=img.shape[0],
        prompt_embeds=encoder_hidden_states,
        num_inference_steps=10,  # Reduced from 50 for faster evaluation
        eta=0.0,  # Deterministic sampling
        output_type="pt",
    ).clamp(-1, 1)  # [B, 2, H, W]
    
    # 💡 FIX: Permute the dimensions of the bm tensor to match the expected format for F.grid_sample
    # from [B, C, H, W] to [B, H, W, C]
    bm_grid = bm.permute(0, 2, 3, 1)
    
    recon = grid_sample(img, bm_grid)  # warp image using predicted bm
    
    # Compute evaluation metrics
    with torch.amp.autocast('cuda'):
        mse_metric = F.mse_loss(recon, img)
        
    # Log metrics and images
    accelerator.log({
        "eval_mse": mse_metric.item(),
        "eval_step": global_step,
    })
    
    # Normalize images from [-1, 1] to [0, 1] for logging
    recon_normalized = (recon + 1.0) / 2.0
    recon_normalized = torch.clamp(recon_normalized, 0.0, 1.0)
    
    log_images_to_wandb(accelerator, recon_normalized, global_step)
    
    pipeline.unet.train()  # Return to training mode
    torch.cuda.empty_cache()

def launch_training():
    """Launch training with proper 4-GPU setup"""
    args = parse_args()
    
    # Automatically set up for 4-GPU training if available
    if torch.cuda.device_count() >= 4 and args.gpu_ids == "0,1,2,3":
        print(f"Detected {torch.cuda.device_count()} GPUs. Setting up 4-GPU training...")
        
        # Set environment variables for optimal 4-GPU performance
        # os.environ['NCCL_TREE_THRESHOLD'] = '0'
        # os.environ['NCCL_ALGO'] = 'Tree'
        # os.environ['NCCL_MIN_NCHANNELS'] = '4'
        # os.environ['NCCL_MAX_NCHANNELS'] = '16'
        
        # Increase batch size for 4-GPU training
        # Increase batch size for 4-GPU training
        # if args.batch_size < 8:
        #     print(f"Increasing batch size from {args.batch_size} to {args.batch_size * 4} for 4-GPU training")
        #     args.batch_size = args.batch_size * 4
            
    elif torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using available GPUs...")
    else:
        print("Single GPU training")
    
    main(args)

if __name__ == "__main__":
    launch_training()
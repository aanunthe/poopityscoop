import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import DDIMScheduler, UNet2DConditionModel
from transformers import CLIPProcessor, CLIPModel
import torch.multiprocessing as mp

# Import components from existing files
from data_loader import Doc3d
from train_ddim import FastDDIMPipeline # Re-use the pipeline from train_ddim
from utils import grid_sample

def tensor_to_image(tensor):
    """Converts a [-1, 1] torch tensor to a [0, 255] numpy image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Move to CPU, convert to numpy, and scale to [0, 255]
    img_np = (tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img_np

def main(args):
    # Set up accelerator for single-GPU inference
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

    # --- 1. Create Output Directories ---
    rec_dir = os.path.join(args.output_dir, "rectified")
    gt_dir = os.path.join(args.output_dir, "ground_truth")
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # --- 2. Load Dataset ---
    print("Loading dataset...")
    # Create a stub args object for Doc3d (only needs image_size attribute)
    class StubArgs:
        pass
    stub_args = StubArgs()
    stub_args.image_size = args.image_size
    dataset = Doc3d(args.dataset_name, stub_args, resize=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, # Use provided batch size
        shuffle=False, # Do not shuffle for evaluation
        num_workers=min(4, mp.cpu_count()),
        pin_memory=True,
    )

    # --- 3. Load Model from Checkpoint ---
    print(f"Loading model from checkpoint: {args.checkpoint_path}")

    # Create the U-Net model
    unet = UNet2DConditionModel(
        sample_size=288,
        in_channels=2,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        encoder_hid_dim=768,
        attention_head_dim=64,
        use_linear_projection=True,
        only_cross_attention=False,
        upcast_attention=False,
    )

    # Try multiple checkpoint formats that Accelerator might use
    possible_paths = [
        os.path.join(args.checkpoint_path, "pytorch_model.bin"),
        os.path.join(args.checkpoint_path, "model.safetensors"),
        os.path.join(args.checkpoint_path, "unet", "diffusion_pytorch_model.bin"),
        os.path.join(args.checkpoint_path, "unet", "diffusion_pytorch_model.safetensors"),
        # Add support for direct save_pretrained output
        os.path.join(args.checkpoint_path, "diffusion_pytorch_model.bin"),
        os.path.join(args.checkpoint_path, "diffusion_pytorch_model.safetensors"),
    ]

    unet_weights_path = None
    for path in possible_paths:
        if os.path.exists(path):
            unet_weights_path = path
            break

    if unet_weights_path is None:
        print(f"Error: Could not find model weights in checkpoint directory: {args.checkpoint_path}")
        print(f"Tried the following paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nCheckpoint directory contents:")
        if os.path.exists(args.checkpoint_path):
            for item in os.listdir(args.checkpoint_path):
                print(f"  - {item}")
        else:
            print(f"  Checkpoint directory does not exist!")
        import sys
        sys.exit(1)

    # Load the weights
    if unet_weights_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(unet_weights_path)
        unet.load_state_dict(state_dict)
    else:
        unet.load_state_dict(torch.load(unet_weights_path, map_location="cpu"))

    print(f"U-Net weights loaded successfully from: {unet_weights_path}")

    # Move U-Net to device before creating pipeline (critical for torch.compile)
    unet = unet.to(device)
    print(f"U-Net moved to device: {device}")

    # Load CLIP model
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    clip_model.eval()

    # Load Scheduler
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

    # --- 4. Prepare Models and Pipeline ---
    pipeline = FastDDIMPipeline(unet, noise_scheduler)
    pipeline, clip_model, dataloader = accelerator.prepare(pipeline, clip_model, dataloader)
    pipeline.unet.eval()

    # --- 5. Run Generation Loop ---
    print("Generating evaluation images...")
    image_counter = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Images"):
            img = batch['img'] # Distorted image, [-1, 1]
            gt_bm = batch['bm'] # Ground truth BM, [-1, 1]


            # --- Get CLIP condition ---
            img_normalized = (img + 1.0) / 2.0 # Normalize to [0, 1] for CLIP
            clip_inputs = processor(images=img_normalized, return_tensors='pt', do_rescale=False)
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            encoder_hidden_states = clip_model.get_image_features(**clip_inputs)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

            # --- Generate Rectified Image (from predicted BM) ---
            pred_bm = pipeline(
                batch_size=img.shape[0],
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=20, # Fast inference
                eta=0.0,
                output_type="pt",
            ).clamp(-1, 1) # [B, 2, H, W]
            
            pred_bm_grid = pred_bm.permute(0, 2, 3, 1) # [B, H, W, 2]
            rectified_img = grid_sample(img, pred_bm_grid)

            # --- Generate Ground Truth "Scan" (from GT BM) ---
            gt_bm_grid = gt_bm.permute(0, 2, 3, 1) # [B, H, W, 2]
            gt_scan_img = grid_sample(img, gt_bm_grid)

            # --- Save images to disk ---
            for i in range(img.shape[0]):
                rec_img_np = tensor_to_image(rectified_img[i:i+1])
                gt_img_np = tensor_to_image(gt_scan_img[i:i+1])
                
                # Use BGR format for cv2.imwrite
                rec_img_bgr = cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR)
                gt_img_bgr = cv2.cvtColor(gt_img_np, cv2.COLOR_RGB2BGR)
                
                save_name = f"{image_counter}.png"
                cv2.imwrite(os.path.join(rec_dir, save_name), rec_img_bgr)
                cv2.imwrite(os.path.join(gt_dir, save_name), gt_img_bgr)
                
                image_counter += 1

                #STEP CAP
                if image_counter >= 150:
                    break

    print(f"Finished generating {image_counter} image pairs in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rectified images from a trained model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint directory (e.g., logs/run_name/checkpoint-1000).')
    parser.add_argument('--dataset_name', type=str, default='./data/doc3d', help='Path to the root of the Doc3D dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated "rectified" and "ground_truth" subfolders.')
    parser.add_argument('--image_size', type=int, default=288, help='Image size to use for generation.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID to use for generation.')
    
    args = parser.parse_args()
    main(args)

# DiffusionDocumentRestoration - Code Index

## Project Overview
This project implements diffusion-based document restoration using the Doc3D dataset. The system learns to predict backward mapping (BM) fields that can be used to unwarp distorted document images back to their flat, readable form.

## Core Architecture
- **Input**: Distorted document images (3 channels: RGB)
- **Output**: Backward mapping fields (2 channels: x,y coordinates)
- **Goal**: Learn to predict BM fields that can unwarp documents using grid sampling

## File Structure & Descriptions

### Configuration Files
- **`requirements.txt`** - Python dependencies including PyTorch, Diffusers, Transformers, OpenCV, etc.
- **`acc_config.yaml`** - Accelerate configuration for multi-GPU training (2 GPUs, mixed precision fp16)

### Data Management
- **`data_loader.py`** - Main data loading module
  - `Doc3d` class: Loads Doc3D dataset with images, backward mappings (BM), world coordinates (WC), and reconstruction masks
  - `wc_norm()`: Normalizes world coordinates using dataset statistics
  - `extract_textline()`: Text line detection and extraction (currently unused)
  - `reverse_bm()`: Applies backward mapping to reconstruct images
  - `Data` class: Alternative dataset loader for WarpDoc dataset

- **`download_doc3d.sh`** - Script to download Doc3D dataset from Stony Brook University
  - Downloads images, world coordinates, backward mappings, and reconstruction masks
  - Currently configured to download first 10 batches (49,990 samples)

### Model Architecture
- **`model.py`** - Neural network architectures
  - `UNetEncoderBlock`: Basic encoder block with Conv2D, BatchNorm, ReLU
  - `UNetEncoder`: Encoder network that processes images to latent representations
  - `UnetDecoder`: Decoder network that generates 2-channel output (BM fields)
  - `EncDec`: Combined encoder-decoder model for direct BM prediction

### Training Scripts
- **`train.py`** - Main DDPM training script
  - Uses `UNet2DConditionModel` with CLIP conditioning
  - Combines ELBO loss (noise prediction) and reconstruction loss
  - Supports multi-GPU training with Accelerate

- **`train_ddim.py`** - DDIM training script
  - Similar to DDPM but uses DDIM scheduler for faster inference
  - Custom `DDIMPipeline` implementation
  - Same loss combination as DDPM

- **`train_encoder.py`** - Direct encoder-decoder training
  - Uses custom `EncDec` model from `model.py`
  - Simple MSE loss between predicted and ground truth BM fields
  - No diffusion process, direct regression

- **`train_unet.py`** - UNet training without conditioning
  - Uses `UNet2DModel` without CLIP conditioning
  - Direct prediction of BM fields
  - Simpler architecture for baseline comparison

### Utilities
- **`utils.py`** - Utility functions and custom pipelines
  - `predict_x0_from_xt()`: Reconstructs clean data from noisy data and predicted noise
  - `grid_sample()`: Applies backward mapping to warp images
  - `DDPMPipeline`: Custom DDPM pipeline for inference
  - `log_images_to_wandb()`: Logs generated images to Weights & Biases
  - `make_grid()`: Creates image grids for visualization

- **`parse_args.py`** - Command line argument parser
  - Training hyperparameters (batch size, learning rate, epochs)
  - Model parameters (image size, alpha for loss weighting)
  - Logging and output configuration

## Data Structure (Doc3D Dataset)
```
doc3d/
├── img/1/          # Document images (PNG files)
├── bm/1/           # Backward mapping fields (MAT files)
├── wc/1/           # World coordinates (EXR files)
└── recon/1/        # Reconstruction masks (PNG files)
```

## Key Components

### 1. Diffusion Models
- **DDPM**: Denoising Diffusion Probabilistic Models for BM field generation
- **DDIM**: Denoising Diffusion Implicit Models for faster sampling
- Both use CLIP image features as conditioning

### 2. Loss Functions
- **ELBO Loss**: Standard diffusion noise prediction loss
- **Reconstruction Loss**: MSE between predicted and ground truth BM fields
- **Combined Loss**: `loss_elbo + alpha * loss_recon`

### 3. Data Processing
- Images normalized to [-1, 1] range
- BM fields normalized to [-1, 1] range
- World coordinates normalized using dataset statistics
- Optional masking using reconstruction data

### 4. Training Infrastructure
- Multi-GPU support via Accelerate
- Mixed precision training (fp16)
- Weights & Biases logging
- Cosine learning rate scheduling with warmup

### 5. GPU Parallelization
- **Configuration**: Set via `--gpu_ids` argument (e.g., "0,1" or "0")
- **Implementation**: Uses Hugging Face Accelerate for distributed training
- **Data Parallel**: Automatic batch splitting across GPUs
- **Gradient Sync**: Gradients averaged across all GPUs
- **Mixed Precision**: FP16 training for memory efficiency

## Usage Patterns

### Training
```bash
# DDPM training (multi-GPU)
python train.py --batch_size 8 --num_epochs 10 --alpha 0.01 --gpu_ids "0,1"

# DDIM training (single GPU)
python train_ddim.py --batch_size 8 --num_epochs 10 --alpha 0.01 --gpu_ids "0"

# Direct encoder-decoder training
python train_encoder.py --batch_size 8 --num_epochs 10 --gpu_ids "0,1"

# UNet baseline training
python train_unet.py --batch_size 8 --num_epochs 10 --gpu_ids "0,1"
```

### Data Download
```bash
bash download_doc3d.sh /path/to/output/directory
```

## Model Variants

1. **Conditioned Diffusion (train.py, train_ddim.py)**
   - Uses CLIP features as conditioning
   - More complex but potentially better quality
   - Requires more computational resources

2. **Direct Regression (train_encoder.py)**
   - Simple encoder-decoder architecture
   - Fast training and inference
   - Good baseline for comparison

3. **Unconditioned UNet (train_unet.py)**
   - Standard UNet without conditioning
   - Simpler architecture
   - Useful for ablation studies

## Key Dependencies
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models library
- **Transformers**: CLIP model for image conditioning
- **Accelerate**: Multi-GPU training
- **OpenCV**: Image processing
- **Weights & Biases**: Experiment tracking

## Notes
- The project focuses on document unwarping using backward mapping fields
- Multiple training approaches are implemented for comparison
- The Doc3D dataset provides comprehensive 3D document information
- All models output 2-channel backward mapping fields for image warping

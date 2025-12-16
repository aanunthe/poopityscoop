import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import torch


from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

def eval_unwarp(rectified_img_gray, gt_img_gray, masked_gt_gray):
    """
    Calculates the Multi-Scale SSIM (MS-SSIM) and Local Distortion (LD).
    """
    
    # --- 1. Multi-Scale SSIM (MS-SSIM) ---
    # Convert numpy to tensor: (B, C, H, W)
    # img1: Distorted/Rectified, img2: Ground Truth
    
    # Normalize to [0, 1] as expected by torchmetrics with data_range=1.0
    img1_t = torch.from_numpy(rectified_img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    img2_t = torch.from_numpy(masked_gt_gray).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Calculate MS-SSIM
    # Move metric to same device as data (if using GPU, usually CPU is fine for evaluation loops)
    if torch.cuda.is_available():
        msssim_metric.to('cuda')
        img1_t = img1_t.to('cuda')
        img2_t = img2_t.to('cuda')

    with torch.no_grad():
        ms_val = msssim_metric(img1_t, img2_t)
        ms = ms_val.item()

    # --- 2. Local Distortion (LD) ---
    # NOTE: We use dense optical flow as a proxy for SIFT Flow.
    
    # Preprocessing: Gaussian Blur
    im1_blur = cv2.GaussianBlur(rectified_img_gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)
    im2_blur = cv2.GaussianBlur(gt_img_gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)

    # Downsample by 0.5 (Scale: 1 -> 0.5)
    im1_small = cv2.resize(im1_blur, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    im2_small = cv2.resize(im2_blur, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # Calculate dense optical flow
    # Using Farneback as a robust alternative to SIFT-Flow
    flow = cv2.calcOpticalFlowFarneback(im1_small, im2_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    vx = flow[..., 0]
    vy = flow[..., 1]

    # Magnitude of displacement
    d = np.sqrt(vx**2 + vy**2)

    # Validate mask
    # We want to check distortion only where the ground truth is valid.
    # Replicate mask preprocessing
    im3_blur = cv2.GaussianBlur(masked_gt_gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)
    im3_small = cv2.resize(im3_blur, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    
    # Valid region: where masked GT is non-zero
    valid_mask = (im3_small != 0)
    
    if np.any(valid_mask):
        ld = np.mean(d[valid_mask])
    else:
        ld = 0.0

    return ms, ld


def main(args):
    """Main evaluation loop."""
    
    print(f"Running evaluation...")
    print(f"Rectified images path: {args.rec_path}")
    print(f"Ground truth path:     {args.gt_path}")

    # Find all rectified images
    rec_image_paths = sorted(glob.glob(os.path.join(args.rec_path, "*.png")))
    
    if not rec_image_paths:
        print("Error: No images found in rec_path.")
        return
        
    total_ms = 0
    total_ld = 0
    valid_images = 0
    
    # Target area from MATLAB script
    target_area = 598400.0

    for rec_path in tqdm(rec_image_paths, desc="Calculating Metrics"):
        img_name = os.path.basename(rec_path)
        gt_path = os.path.join(args.gt_path, img_name)
        
        # Handle potential mismatch in naming if strictly required
        if not os.path.exists(gt_path):
            print(f"Warning: Missing corresponding GT for {img_name}")
            continue

        # Load images
        A1_color = cv2.imread(rec_path)
        ref_color = cv2.imread(gt_path)

        if A1_color is None or ref_color is None:
            print(f"Warning: Could not read {img_name}")
            continue

        # Convert to grayscale
        A1 = cv2.cvtColor(A1_color, cv2.COLOR_BGR2GRAY)
        ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

        # --- Replicate MATLAB resizing logic ---
        h, w = ref.shape
        b = np.sqrt(target_area / (h * w))
        
        new_h, new_w = int(np.round(h * b)), int(np.round(w * b))
        
        # Resize GT
        ref_resized = cv2.resize(ref, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize Rectified to match GT
        A1_resized = cv2.resize(A1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # --- Replicate MATLAB masking logic ---
        # m1 = A1 == 0;
        mask = (A1_resized == 0)
        
        # ref_msk = ref; ref_msk(m1) = 0;
        ref_msk_resized = ref_resized.copy()
        ref_msk_resized[mask] = 0
        
        # --- Calculate metrics ---
        ms_1, ld_1 = eval_unwarp(A1_resized, ref_resized, ref_msk_resized)
        
        total_ms += ms_1
        total_ld += ld_1
        valid_images += 1

    if valid_images > 0:
        avg_ms = total_ms / valid_images
        avg_ld = total_ld / valid_images
        
        print("\n--- Evaluation Results ---")
        print(f"Processed {valid_images} images.")
        print(f"Average MS-SSIM: {avg_ms:.6f}")
        print(f"Average Local Distortion (LD): {avg_ld:.6f}")
    else:
        print("No valid image pairs were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation metrics (MS-SSIM, LD) on generated images.")
    parser.add_argument('--rec_path', type=str, required=True, help='Path to the directory of rectified (unwarped) images.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the directory of ground truth (scan) images.')
    
    args = parser.parse_args()
    main(args)
import cv2
import numpy as np
import os
import glob
import argparse
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def eval_unwarp(rectified_img_gray, gt_img_gray, masked_gt_gray):
    """
    Calculates the Multi-Scale SSIM (MS-SSIM) and Local Distortion (LD).
    
    NOTE: The original MATLAB code uses 'mexDenseSIFT' and 'SIFTflowc2f',
    which are proprietary functions. We are substituting this with
    OpenCV's standard dense optical flow algorithm, 'calcOpticalFlowFarneback'.
    This is a well-established alternative for calculating dense displacement fields.
    """
    
    # --- 1. Multi-Scale SSIM (MS-SSIM) ---
    # Replicate the 5-level pyramid from the MATLAB code
    wt = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    ss_scores = []
    
    # Use float64 for precision, matching MATLAB's im2double
    x = rectified_img_gray.astype(np.float64) / 255.0
    z = masked_gt_gray.astype(np.float64) / 255.0

    for i in range(5):
        # K1=0.01, K2=0.03 are SSIM defaults, matching MATLAB's ssim()
        # win_size=11 is also a common default.
        s = ssim(x, z, data_range=1.0, win_size=11, K1=0.01, K2=0.03, gaussian_weights=True)
        ss_scores.append(s)
        
        # Replicate impyramid(x, 'reduce') - downscale by 2
        # Using linear interpolation as a standard downscaling filter
        x = cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        z = cv2.resize(z, (z.shape[1] // 2, z.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    ms = np.dot(wt, ss_scores)

    # --- 2. Local Distortion (LD) ---
    # Replicate MATLAB's preprocessing for SIFT Flow
    # imfilter(x,fspecial('gaussian',7,1.),'same','replicate')
    im1_blur = cv2.GaussianBlur(rectified_img_gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)
    im2_blur = cv2.GaussianBlur(gt_img_gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)

    # imresize(..., 0.5, 'bicubic')
    im1_small = cv2.resize(im1_blur, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    im2_small = cv2.resize(im2_blur, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # Calculate dense optical flow (Farneback) as a substitute for SIFT Flow
    # Parameters are common defaults
    flow = cv2.calcOpticalFlowFarneback(im1_small, im2_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    vx = flow[..., 0]
    vy = flow[..., 1]

    # Calculate magnitude of displacement
    d = np.sqrt(vx**2 + vy**2)

    # Create the mask for valid regions
    # Replicates imresize(imfilter(z, ...), 0.5)
    im3_blur = cv2.GaussianBlur(masked_gt_gray, (7, 7), 1, borderType=cv2.BORDER_REPLICATE)
    im3_small = cv2.resize(im3_blur, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    
    # mskk = (im3==0)
    # We care about where the *masked ground truth* is NOT zero
    valid_mask = (im3_small != 0)
    
    # ld = mean(d(~mskk))
    if np.any(valid_mask):
        ld = np.mean(d[valid_mask])
    else:
        ld = 0.0 # Avoid mean of empty slice warning

    return ms, ld


def main(args):
    """Main evaluation loop."""
    
    print(f"Running evaluation...")
    print(f"Rectified images path: {args.rec_path}")
    print(f"Ground truth path:     {args.gt_path}")

    # Find all rectified images
    rec_image_paths = sorted(glob.glob(os.path.join(args.rec_path, "*.png")))
    gt_image_paths = sorted(glob.glob(os.path.join(args.gt_path, "*.png")))

    if not rec_image_paths or not gt_image_paths:
        print("Error: No images found in one or both directories.")
        return

    if len(rec_image_paths) != len(gt_image_paths):
        print(f"Warning: Mismatch in image counts. Rectified: {len(rec_image_paths)}, GT: {len(gt_image_paths)}")
        
    total_ms = 0
    total_ld = 0
    valid_images = 0
    
    # Target area from MATLAB script
    target_area = 598400.0

    for rec_path in tqdm(rec_image_paths, desc="Calculating Metrics"):
        img_name = os.path.basename(rec_path)
        gt_path = os.path.join(args.gt_path, img_name)
        
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

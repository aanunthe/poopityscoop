import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F

# --- MS-SSIM Implementation (PyTorch) ---
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        # For MS-SSIM, we need the contrast (structure) component as well as the full SSIM
        # VIF implementation details or standard Wang et al. 
        # MCS = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        # But for the last scale, we use the full SSIM.
        
        mcs_map = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        if size_average:
            return ssim_map.mean(), mcs_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1), mcs_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # Weights for MS-SSIM (Wang et al. 2003)
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
        
        levels = weights.size()[0]
        mssim = []
        mcs = []
        
        # Ensure window is on the same device
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        for i in range(levels):
            ssim_val, mcs_val = self._ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)
            mssim.append(ssim_val)
            mcs.append(mcs_val)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        # Formula: L_M * Product(MCS_j ^ beta_j) * Product(S_j ^ gamma_j) ? 
        # Standard implementation (e.g. pytorch-msssim):
        # MS-SSIM = (SSIM_M)^beta_M * Product_{j=1..M-1} (MCS_j)^beta_j
        # where exponents are the weights.
        
        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        
        # Only the last scale SSIM is used for the luminance component (approximated by SSIM logic)
        # In many implementations: result = Prod(MCS[i]^weights[i] for i in 0..M-2) * (SSIM[M-1]^weights[M-1])
        
        # Let's strictly follow the standard:
        # P = Product_{j=1}^{M} (MCS_j)^{\beta_j} --- wait, usually it is: 
        # MS-SSIM = (L_M)^alpha_M * Product_{j=1}^{M} (C_j)^beta_j (S_j)^gamma_j
        # Approximations simplify this. 
        
        # Using the logic from 'pytorch-msssim' library (VainF):
        # output = prod(mcs[:levels-1] ** weights[:levels-1]) * (mssim[levels-1] ** weights[levels-1])
        
        pow1 = mcs ** weights
        pow2 = mssim ** weights
        
        # NOTE: mssim list contains SSIM at each scale.
        # But definition says we only use full SSIM at the finest scale (or coarsest in the pyramid?)
        # Wang's paper: "We calculate SSIM at scale M... and contrast/structure at other scales"
        # Scale 1 is original, Scale 5 is coarsest.
        # Loop i=0 is scale 1. i=4 is scale 5 (coarsest).
        
        # Common implementations compute the weighted product of all MCS and the SSIM of the LAST scale.
        
        output = torch.prod(mcs[0:levels-1] ** weights[0:levels-1]) * (mssim[levels-1] ** weights[levels-1])
        return output

# Initialize metrics globally to avoid recreation
msssim_metric = MSSSIM(channel=1) # We use grayscale

def eval_unwarp(rectified_img_gray, gt_img_gray, masked_gt_gray):
    """
    Calculates the Multi-Scale SSIM (MS-SSIM) and Local Distortion (LD).
    """
    
    # --- 1. Multi-Scale SSIM (MS-SSIM) ---
    # Convert numpy to tensor
    # img1: Distorted/Rectified, img2: Ground Truth
    
    img1_t = torch.from_numpy(rectified_img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    img2_t = torch.from_numpy(masked_gt_gray).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Calculate MS-SSIM
    with torch.no_grad():
        ms_val = msssim_metric(img1_t, img2_t)
        ms = ms_val.item()

    # --- 2. Local Distortion (LD) ---
    # NOTE: We use dense optical flow as a proxy for SIFT Flow (proprietary).
    
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
    gt_image_paths = sorted(glob.glob(os.path.join(args.gt_path, "*.png")))

    if not rec_image_paths or not gt_image_paths:
        print("Error: No images found in one or both directories.")
        return
        
    total_ms = 0
    total_ld = 0
    valid_images = 0
    
    # Target area from MATLAB script
    target_area = 598400.0

    for rec_path in tqdm(rec_image_paths, desc="Calculating Metrics"):
        img_name = os.path.basename(rec_path)
        gt_path = os.path.join(args.gt_path, img_name)
        
        # Handle potential mismatch in naming if strictly required, but usually they match
        if not os.path.exists(gt_path):
            # Try to find loose match if needed or just skip
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

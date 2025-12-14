import numpy as np
import cv2
import torch
from evaluation_metrics import eval_unwarp

def test_metrics():
    print("Testing Evaluation Metrics...")
    
    # Create dummy images (grayscale)
    h, w = 256, 256
    
    # 1. Perfect match
    img1 = np.ones((h, w), dtype=np.uint8) * 128
    img2 = np.ones((h, w), dtype=np.uint8) * 128
    mask = np.ones((h, w), dtype=np.uint8) * 128
    
    ms, ld = eval_unwarp(img1, img2, mask)
    print(f"Perfect Match -> MS-SSIM: {ms:.4f} (Expected ~1.0), LD: {ld:.4f} (Expected ~0.0)")
    
    # 2. Slight difference
    img1_diff = img1.copy()
    cv2.circle(img1_diff, (100, 100), 20, 0, -1) # meaningful structure change
    
    ms, ld = eval_unwarp(img1_diff, img2, mask)
    print(f"Slight Diff   -> MS-SSIM: {ms:.4f}, LD: {ld:.4f}")
    
    # 3. Geometric distortion (shift) -> should trigger LD
    # Create a pattern to track flow
    img_pattern = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, 20):
        cv2.line(img_pattern, (0, i), (w, i), 255, 2)
    for i in range(0, w, 20):
        cv2.line(img_pattern, (i, 0), (i, h), 255, 2)
        
    M = np.float32([[1, 0, 5], [0, 1, 5]]) # Shift by 5 pixels
    img_shifted = cv2.warpAffine(img_pattern, M, (w, h))
    
    ms, ld = eval_unwarp(img_shifted, img_pattern, img_pattern)
    print(f"Shifted (5px) -> MS-SSIM: {ms:.4f}, LD: {ld:.4f}")

if __name__ == "__main__":
    test_metrics()

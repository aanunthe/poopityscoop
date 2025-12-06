import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['WANDB_MODE'] = 'offline'
os.environ['NCCL_P2P_DISABLE'] = '1'
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import log_images_to_wandb, grid_sample, predict_x0_from_xt
import datetime
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm 
import torch
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
from model import EncDec
import torchvision.models as models
from torchvision.models import VGG19_Weights

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use VGG19 for perceptual loss
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_layers = torch.nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:18],  # relu3_4
            vgg[:27],  # relu4_4
        ])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        loss = 0.0
        for vgg_layer in self.vgg_layers:
            pred_feat = vgg_layer(pred)
            target_feat = vgg_layer(target)
            loss += F.mse_loss(pred_feat, target_feat)
        return loss

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # Convert to grayscale if needed
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        # Compute edges
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return F.mse_loss(pred_edges, target_edges)

def main(args):
    # Set GPU visibility before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(args.output_dir, args.run_name),
        automatic_checkpoint_naming=False,
        total_limit=None
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_config,
        log_with='wandb',
        mixed_precision='fp16'
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
    
    dataset = Doc3d(args.dataset_name, args, resize=True)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size 
    )

    model = EncDec(1024)
    
    # Initialize loss functions
    perceptual_loss = PerceptualLoss()
    edge_loss = EdgeLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4  # Add weight decay for regularization
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    model, optimizer, train_dataloader, lr_scheduler, perceptual_loss, edge_loss = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, perceptual_loss, edge_loss
    )
    
    global_step = 0
    for epoch in tqdm(range(args.num_epochs), leave=False, disable=not accelerator.is_main_process, desc='Epochs'):
        for step, batch in enumerate(tqdm(train_dataloader, leave=False, disable=not accelerator.is_main_process, desc='Data')):
            img = batch['img']
            bm = batch['bm']
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    bm_pred = model(img)
                    
                    # Compute reconstruction for perceptual loss
                    recon_pred = grid_sample(img, bm_pred.permute(0, 2, 3, 1))
                    recon_target = grid_sample(img, bm.permute(0, 2, 3, 1))
                    
                    # Combined loss function
                    mse_loss = F.mse_loss(bm_pred, bm)
                    perc_loss = perceptual_loss(recon_pred, recon_target) * 0.1
                    edge_loss_val = edge_loss(recon_pred, recon_target) * 0.05
                    
                    loss = mse_loss + perc_loss + edge_loss_val
                    
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)  # Reduced clip value
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
            
            if accelerator.is_main_process and global_step % args.log_freq == 0:
                accelerator.log({
                    "total_loss": loss.item(),
                    "mse_loss": mse_loss.item(),
                    "perceptual_loss": perc_loss.item(),
                    "edge_loss": edge_loss_val.item(),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr']
                })
            
            if accelerator.is_main_process and global_step % args.save_freq == 0:
                evaluate(batch['img'], accelerator.unwrap_model(model), accelerator, global_step)
        accelerator.wait_for_everyone()



def compute_ssim(img1, img2):
    """Compute SSIM between two images"""
    C1 = 0.01**2
    C2 = 0.03**2
    
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

@torch.no_grad()
def evaluate(img, model, accelerator, global_step):   
    bm = model(img)
    recon = grid_sample(img, bm.permute(0, 2, 3, 1))
    
    # Compute evaluation metrics
    ssim_score = compute_ssim(recon, img)
    psnr_score = compute_psnr(recon, img)
    
    # Log metrics
    accelerator.log({
        "eval_ssim": ssim_score.item(),
        "eval_psnr": psnr_score.item(),
    })
    
    # Normalize images from [-1, 1] to [0, 1] for logging
    recon_normalized = (recon + 1.0) / 2.0
    recon_normalized = torch.clamp(recon_normalized, 0.0, 1.0)
    
    log_images_to_wandb(
        accelerator, recon_normalized, global_step
    )  

if __name__ == "__main__":
    args = parse_args()
    main(args) 
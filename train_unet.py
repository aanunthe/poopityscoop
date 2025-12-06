import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['WANDB_MODE'] = 'online'
os.environ['NCCL_P2P_DISABLE'] = '1'
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import log_images_to_wandb, DDPMPipeline, grid_sample, predict_x0_from_xt
import datetime
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm 
import torch
from torch.optim import AdamW
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from parse_args import parse_args
import torch
import torch.nn.functional as F
from data_loader import Doc3d
from transformers import CLIPProcessor, CLIPModel

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
    
    #dataset = Data('./data/WarpDoc', transform=transform)
    dataset = Doc3d(args.dataset_name, args, resize=True)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size 
    )

    model = UNet2DModel(
        sample_size=288,
        in_channels=3,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),  
    )
    

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    for epoch in tqdm(range(args.num_epochs), leave=False, disable=not accelerator.is_main_process, desc='Epochs'):
        for step, batch in enumerate(tqdm(train_dataloader, leave=False, disable=not accelerator.is_main_process, desc='Data')):
            img = batch['img']
            true_bm = batch['bm']
        
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    B = img.shape[0]
                    t = torch.zeros(B, dtype=torch.long, device=img.device)
                    pred_bm = model(img, timestep=t).sample
                    
                    #recon = grid_sample(batch['img'], bm_pred.permute(0, 2, 3, 1))
                    #true_img = grid_sample(batch['img'], batch['bm'].permute(0, 2, 3, 1))
                loss = F.mse_loss(pred_bm, true_bm)                    
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
            
            if accelerator.is_main_process and global_step % args.log_freq == 0:
                accelerator.log({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr']
                })

            if accelerator.is_main_process:
                if global_step % args.save_freq == 0:
                    with torch.no_grad():
                        evaluate(batch['img'], accelerator.unwrap_model(model), accelerator, global_step)
            accelerator.wait_for_everyone()

def evaluate(img, model, accelerator, global_step):   
    B = img.shape[0]
    t = torch.zeros(B, dtype=torch.long, device=img.device)
    bm = model(img, timestep=t).sample
    recon = grid_sample(img, bm.permute(0, 2, 3, 1))
    
    # Normalize images from [-1, 1] to [0, 1] for logging
    recon_normalized = (recon + 1.0) / 2.0
    recon_normalized = torch.clamp(recon_normalized, 0.0, 1.0)
    
    log_images_to_wandb(
        accelerator, recon_normalized, global_step
    )  
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
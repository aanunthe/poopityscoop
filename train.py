import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['WANDB_MODE'] = 'offline'
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
from diffusers import DDPMScheduler, UNet2DConditionModel
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

    model = UNet2DConditionModel(
        sample_size=288,
        in_channels=2,
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
        encoder_hid_dim=768,    
    )
    
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    for epoch in tqdm(range(args.num_epochs), leave=False, disable=not accelerator.is_main_process, desc='Epochs'):
        for step, batch in enumerate(tqdm(train_dataloader, leave=False, disable=not accelerator.is_main_process, desc='Data')):
            img = batch['img']
            x = processor(images=batch['img'], return_tensors='pt', do_rescale=False)
            x = {k: v.to(device) for k, v in x.items()}
            with torch.no_grad():
                x = clip_model.get_image_features(**x).unsqueeze(1)
            latent = batch['bm']
            noise = torch.randn(latent.shape).to(device)
            bs = latent.shape[0]
            
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latent.device
            ).long()
            
            noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    noise_pred = model(
                        noisy_latent,
                        timestep=timesteps,
                        encoder_hidden_states=x,
                    ).sample
                    
                    bm_pred = predict_x0_from_xt(noisy_latent, timesteps, noise_pred, noise_scheduler).clamp(-1, 1)
                    
                    #recon = grid_sample(batch['img'], bm_pred.permute(0, 2, 3, 1))
                    #true_img = grid_sample(batch['img'], batch['bm'].permute(0, 2, 3, 1))
                    loss_recon = F.mse_loss(bm_pred, batch['bm'])
                    loss_elbo = F.mse_loss(noise_pred, noise)                    
                    loss = loss_elbo + args.alpha * loss_recon
                    
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
                    "loss_elbo": loss_elbo.item(),
                    "loss_recon": loss_recon.item(),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr']
                })
            
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(accelerator.unwrap_model(model), noise_scheduler)
                if global_step % args.save_freq == 0:
                    evaluate(args, epoch, x, batch['img'], pipeline, accelerator, global_step)
            accelerator.wait_for_everyone()

def evaluate(args, epoch, x, img, pipeline,  accelerator, global_step):   
    bm = pipeline(x).clamp(-1, 1)
    recon = grid_sample(img, bm)
    
    # Normalize images from [-1, 1] to [0, 1] for logging
    recon_normalized = (recon + 1.0) / 2.0
    recon_normalized = torch.clamp(recon_normalized, 0.0, 1.0)
    
    log_images_to_wandb(
        accelerator, recon_normalized, global_step
    )  
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
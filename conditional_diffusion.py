# conditional_diffusion_v4.0.py

import os
import math
import sys
import json
import argparse
import warnings
from glob import glob
import numpy as np
from PIL import Image
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from diffusion_model import DiffUNet

# Suppress harmless UserWarning from PyTorch scheduler
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The epoch parameter in `scheduler.step()` was not necessary and is being deprecated.*"
)

# -----------------------------
# Configuration via Argparse
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Conditional Diffusion Model Training v4.0 with DiffUNet")
    
    g_paths = parser.add_argument_group('Paths')
    g_paths.add_argument("--dataset_path", type=str, default="C:/Users/user/Desktop/U23EC026/Food_Dataset", help="Path to the root of your image dataset.")
    g_paths.add_argument("--output_dir", type=str, default="C:/Users/user/Desktop/U23EC026/Outputs", help="Directory to save outputs and checkpoints.")
    g_paths.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")

    g_train = parser.add_argument_group('Training')
    g_train.add_argument("--total_epochs", type=int, default=500, help="Total number of epochs to train for.")
    g_train.add_argument("--batch_size", type=int, default=16, help="Batch size. DiffUNet is large, adjust based on VRAM.")
    g_train.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    g_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    g_train.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs.")
    g_train.add_argument("--save_every_epoch", action=argparse.BooleanOptionalAction, default=True, help="Save a checkpoint and sample grid after each epoch.")

    g_model = parser.add_argument_group('Model')
    g_model.add_argument("--img_size", type=int, default=256, help="Resize images to this size.")
    g_model.add_argument("--in_channels", type=int, default=3, help="Number of input channels (3 for RGB).")
    g_model.add_argument("--num_classes", type=int, default=15, help="Number of classes in the dataset.")
    g_model.add_argument("--large_model", action=argparse.BooleanOptionalAction, default=False, help="Use the large DiffUNet model (requires more VRAM).")

    g_diff = parser.add_argument_group('Diffusion')
    g_diff.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps.")
    g_diff.add_argument("--beta_start", type=float, default=1e-4, help="Starting value of beta for the noise schedule.")
    g_diff.add_argument("--beta_end", type=float, default=0.02, help="Ending value of beta for the noise schedule.")

    g_guid = parser.add_argument_group('Guidance & EMA')
    g_guid.add_argument("--cfg_scale", type=float, default=7.5, help="Scale for Classifier-Free Guidance during sampling.")
    g_guid.add_argument("--uncond_prob", type=float, default=0.1, help="Probability of using an unconditional label during training (for CFG).")
    g_guid.add_argument("--ema_decay", type=float, default=0.9999, help="Target decay rate for EMA.")
    g_guid.add_argument("--ema_base_decay", type=float, default=0.99, help="Starting EMA decay for warmup.")
    g_guid.add_argument("--ema_warmup_epochs", type=int, default=10, help="Number of epochs to warm EMA decay from base to target.")

    g_misc = parser.add_argument_group('Miscellaneous')
    g_misc.add_argument("--sampling_timesteps", type=int, default=50, help="Number of DDIM sampling steps for previews.")
    g_misc.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True, help="Use channels_last memory format for speed on CUDA.")
    g_misc.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

# -----------------------------
# Diffusion schedule, Helper
# -----------------------------
def get_diffusion_schedule(beta_start, beta_end, timesteps, device):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }

def gather_1d(arr: torch.Tensor, t: torch.Tensor):
    return arr[t].view(-1, 1, 1, 1)

# -----------------------------
# Dataset and Dataloader
# -----------------------------
class CustomRGBDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir, self.transform = root_dir, transform
        self.image_files = sorted(glob(os.path.join(root_dir, "**", "*.*"), recursive=True))
        self.image_files = [f for f in self.image_files if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
        
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()]) if os.path.exists(root_dir) else []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if not self.image_files:
            raise RuntimeError(f"FATAL: No images found in {root_dir}. Cannot train.")
        else:
            self.labels = [self.class_to_idx[os.path.basename(os.path.dirname(p))] for p in self.image_files]
            print(f"Found {len(self.image_files)} images belonging to {len(self.classes)} classes.")

    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path, label = self.image_files[idx], self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*args.in_channels, std=[0.5]*args.in_channels)
    ])
    dataset = CustomRGBDataset(root_dir=args.dataset_path, transform=transform)
    if len(dataset.classes) > 0 and args.num_classes != len(dataset.classes):
         raise ValueError(f"Config expects num_classes={args.num_classes}, but found {len(dataset.classes)} directories/classes.")
    # Set num_workers=0 to fix duplicated output on Windows
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

# -----------------------------
# EMA (Exponential Moving Average)
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (self.decay * self.shadow[n] + (1. - self.decay) * p.detach()).clone()
    def state_dict(self): return {"decay": self.decay, "shadow": self.shadow}
    def load_state_dict(self, state): self.shadow, self.decay = state["shadow"], state.get("decay", self.decay)
    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        m = model.module if isinstance(model, nn.DataParallel) else model
        for n, p in m.named_parameters():
            if p.requires_grad: p.data.copy_(self.shadow[n])

# -----------------------------
# Checkpointing and Evaluation
# -----------------------------
def save_checkpoint(epoch, global_step, model, optimizer, scheduler, scaler, ema, output_dir, loss, is_best):
    state = {
        'epoch': epoch, 'global_step': global_step,
        'model_state': (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'scaler_state': scaler.state_dict() if scaler else None,
        'ema_state': ema.state_dict() if ema else None, 'loss': loss
    }
    torch.save(state, os.path.join(output_dir, 'latest_checkpoint.pth'))
    if is_best:
        torch.save(state, os.path.join(output_dir, 'best_checkpoint.pth'))
        print(f"✅ New best checkpoint saved at epoch {epoch+1} with loss {loss:.6f}")

def evaluate_and_save(epoch, global_step, model, optimizer, scheduler, scaler, ema, diff_schedule, args, device, loss, best_loss):
    is_best = loss < best_loss[0]
    if is_best: best_loss[0] = loss
    save_checkpoint(epoch, global_step, model, optimizer, scheduler, scaler, ema, args.output_dir, loss, is_best)
    try:
        ema.apply_shadow(model)
        n_samp = min(args.num_classes, 16)
        sample_labels = torch.arange(n_samp, device=device)
        generated = sample_ddim_conditional(
            model, diff_schedule, n_samples=n_samp, labels=sample_labels,
            img_size=args.img_size, in_channels=args.in_channels, device=device,
            cfg_scale=args.cfg_scale, sampling_timesteps=args.sampling_timesteps, num_classes=args.num_classes
        )
        grid = make_grid(generated, nrow=int(math.sqrt(n_samp)))
        wandb.log({"generated_samples": wandb.Image(grid)})
        save_image(grid, os.path.join(args.output_dir, f'generated_epoch_{epoch+1}.png'))
        print(f"Saved sample grid for epoch {epoch+1}")
    except Exception as e:
        print(f"Sampling after epoch {epoch+1} failed: {e}")
    finally:
        model.train()

# -----------------------------
# UPDATED Sampling Function for DiffUNet
# -----------------------------
@torch.inference_mode()
def sample_ddim_conditional(model, diff_schedule, n_samples, labels, img_size, in_channels, device, num_classes, total_steps=1000, sampling_timesteps=50, cfg_scale=7.5, eta=0.0):
    model.eval()
    x = torch.randn(n_samples, in_channels, img_size, img_size, device=device)
    times = list(reversed(torch.linspace(-1, total_steps - 1, steps=sampling_timesteps + 1).int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    for t, t_prev in tqdm(time_pairs, desc="Conditional DDIM Sampling"):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # AFTER (THE FIX)
        with torch.amp.autocast('cuda', enabled=False):
            # Use the 'y' keyword and type_t for the model call
            pred_output_cond = model(x, t_tensor, y=labels.to(device), type_t="timestep")
            
            uncond_labels = torch.full_like(labels, num_classes).to(device)
            pred_output_uncond = model(x, t_tensor, y=uncond_labels, type_t="timestep")
        
        # Get just the noise prediction (first 3 channels) from the model's output
        pred_noise_cond = pred_output_cond[:, :3, ...]
        pred_noise_uncond = pred_output_uncond[:, :3, ...]
        
        predicted_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        
        alpha_t = diff_schedule["alphas_cumprod"][t] if t >= 0 else torch.tensor(1.0, device=device)
        alpha_t_prev = diff_schedule["alphas_cumprod"][t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) if t > 0 else 0.0
        pred_x0 = (x - torch.sqrt(1. - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_x0.clamp_(-1., 1.)
        pred_dir_xt = torch.sqrt(torch.clamp(1. - alpha_t_prev - sigma_t**2, min=0.0)) * predicted_noise
        x = torch.sqrt(alpha_t_prev) * pred_x0 + pred_dir_xt + sigma_t * torch.randn_like(x)
        
    return (x.clamp(-1., 1.) + 1.) / 2.

# -----------------------------
# Main Training Logic
# -----------------------------
if __name__ == "__main__":

    args = parse_args()

    wandb.init(
        project="conditional-diffusion-food", # Name of your project
        name=f"run_{args.img_size}px_lr{args.lr}", # A specific name for this run
        config=vars(args) # Save all hyperparameters
    )

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device); torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True); os.makedirs(args.dataset_path, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args_v4.0.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    diff_schedule = get_diffusion_schedule(args.beta_start, args.beta_end, args.timesteps, device)
    loader = get_dataloader(args)
    
    # --- UPDATED MODEL INSTANTIATION ---
    print("Initializing powerful DiffUNet with attention...")
    model = DiffUNet(
        in_channels=args.in_channels,
        out_channels=args.in_channels, # This is handled internally by DiffUNet
        num_classes=args.num_classes, # This enables conditional training
        large_model=args.large_model,
        pretrained=None # Training from scratch
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model.to(device)
    if args.channels_last and device.type == 'cuda':
        model.to(memory_format=torch.channels_last); print("Using channels_last memory format.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=args.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.total_epochs - args.warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    ema = EMA(model.module if isinstance(model, nn.DataParallel) else model, decay=args.ema_base_decay)

    start_epoch, global_step = 0, 0
    best_loss = [float("inf")]

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.module.load_state_dict(ckpt['model_state']) if isinstance(model, nn.DataParallel) else model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        
        if ckpt.get('scheduler_state'): scheduler.load_state_dict(ckpt['scheduler_state'])
        
        if ckpt.get('scaler_state'): scaler.load_state_dict(ckpt['scaler_state'])
        if ckpt.get('ema_state'): ema.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_loss[0] = ckpt.get('loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, global step {global_step}. Best loss: {best_loss[0]:.6f}")

    num_train_steps = len(loader) * args.total_epochs
    num_ema_warmup_steps = len(loader) * args.ema_warmup_epochs

    for epoch in range(start_epoch, args.total_epochs):
        model.train(); total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.total_epochs}")
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.channels_last: x = x.to(memory_format=torch.channels_last)
            labels_training = y.clone()
            labels_training[torch.rand(labels_training.size(0), device=device) < args.uncond_prob] = args.num_classes
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = (gather_1d(diff_schedule["sqrt_alphas_cumprod"], t) * x +
                   gather_1d(diff_schedule["sqrt_one_minus_alphas_cumprod"], t) * noise)
            
            # --- UPDATED FORWARD PASS ---
            # AFTER (THE FIX)
            with torch.amp.autocast('cuda', enabled=False):
                predicted_output = model(x_t, t, y=labels_training, type_t="timestep")
                predicted_noise = predicted_output[:, :3, ...] # Slice the first 3 channels
                loss = F.mse_loss(predicted_noise, noise)

            loss_val = loss.item()
            total_loss += loss_val
            scaler.scale(loss / args.accum_steps).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if global_step < num_ema_warmup_steps:
                    ema.decay = args.ema_base_decay + (args.ema_decay - args.ema_base_decay) * (global_step / num_ema_warmup_steps)
                else:
                    ema.decay = args.ema_decay
                ema.update(model.module if isinstance(model, nn.DataParallel) else model)
                global_step += 1
            pbar.set_postfix(loss=loss_val, ema_decay=f"{ema.decay:.5f}")
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6e}")

        wandb.log({"avg_loss": avg_loss, "learning_rate": current_lr, "epoch": epoch + 1})
        
        if args.save_every_epoch:
            evaluate_and_save(epoch, global_step, model, optimizer, scheduler, scaler, ema, diff_schedule, args, device, avg_loss, best_loss)
    
    wandb.finish()
    print(f"Training complete. Best Loss achieved: {best_loss[0]:.6f}")
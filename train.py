import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model import LightDenoisingAutoencoder
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import Config
from data.dataset import create_data_loaders
from models.ddpm import DDPM
from models.unet import UNet

def save_images(clean, noisy, denoised, epoch, save_dir, prefix="train"):
    """Save a grid of images for visualization"""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(4):
        axes[0, i].imshow(clean[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Clean')
        
        axes[1, i].imshow(noisy[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Noisy')
        
        axes[2, i].imshow(denoised[i, 0].cpu(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title('Denoised')
    
    plt.tight_layout()
    save_path = Path(save_dir) / f"{prefix}_samples_epoch_{epoch}.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

def save_comparison_image(noisy_img, clean_img, denoised_img, epoch, save_dir, idx):
    """Save a comparison of noisy, clean, and denoised images"""
    plt.figure(figsize=(15, 5))
    
    # Convert tensors to numpy arrays and squeeze channel dimension
    noisy_img = noisy_img.detach().cpu().squeeze().numpy()
    clean_img = clean_img.detach().cpu().squeeze().numpy()
    denoised_img = denoised_img.detach().cpu().squeeze().numpy()
    
    # Calculate PSNR and SSIM
    noisy_psnr = psnr(clean_img, noisy_img, data_range=1.0)
    denoised_psnr = psnr(clean_img, denoised_img, data_range=1.0)
    noisy_ssim = ssim(clean_img, noisy_img, data_range=1.0)
    denoised_ssim = ssim(clean_img, denoised_img, data_range=1.0)
    
    # Plot images
    plt.subplot(131)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f'Noisy\nPSNR: {noisy_psnr:.2f}\nSSIM: {noisy_ssim:.2f}')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(clean_img, cmap='gray')
    plt.title('Clean')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(denoised_img, cmap='gray')
    plt.title(f'Denoised\nPSNR: {denoised_psnr:.2f}\nSSIM: {denoised_ssim:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = save_dir / f'comparison_epoch_{epoch}_sample_{idx}.png'
    plt.savefig(save_path)
    plt.close()
    
    return save_path, {
        'noisy_psnr': noisy_psnr,
        'denoised_psnr': denoised_psnr,
        'noisy_ssim': noisy_ssim,
        'denoised_ssim': denoised_ssim
    }

def evaluate_model(model, val_loader, device):
    """Evaluate model performance on validation set"""
    model.eval()
    metrics = {
        'psnr': [],
        'ssim': []
    }
    
    with torch.no_grad():
        for noisy_imgs, clean_imgs in val_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            denoised_imgs = model(noisy_imgs)
            
            # Calculate metrics for each image in batch
            for i in range(clean_imgs.size(0)):
                clean = clean_imgs[i].cpu().squeeze().numpy()
                denoised = denoised_imgs[i].cpu().squeeze().numpy()
                
                metrics['psnr'].append(psnr(clean, denoised, data_range=1.0))
                metrics['ssim'].append(ssim(clean, denoised, data_range=1.0))
    
    return {
        'avg_psnr': np.mean(metrics['psnr']),
        'avg_ssim': np.mean(metrics['ssim'])
    }

def train_autoencoder(config):
    """Training loop for autoencoder model"""
    device = torch.device(config.DEVICE)
    
    # Create model
    model = UNet(
        in_channels=1,
        out_channels=1,
        features=config.UNET_CHANNELS,
        attention_layers=config.UNET_ATTENTION_LAYERS,
    ).to(device)
    
    # Create optimizers
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize wandb
    wandb.init(project=config.WANDB_PROJECT, name="autoencoder")
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}") as pbar:
            for batch in pbar:
                clean = batch["clean"].to(device)
                noisy = batch["noisy"].to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    denoised = model(noisy, torch.zeros(noisy.shape[0], device=device))
                    loss = nn.MSELoss()(denoised, clean)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                noisy = batch["noisy"].to(device)
                denoised = model(noisy, torch.zeros(noisy.shape[0], device=device))
                val_loss += nn.MSELoss()(denoised, clean).item()
        
        val_loss /= len(val_loader)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        # Save sample images
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                clean = sample_batch["clean"].to(device)
                noisy = sample_batch["noisy"].to(device)
                denoised = model(noisy, torch.zeros(noisy.shape[0], device=device))
                
                img_path = save_images(clean, noisy, denoised, epoch + 1, 
                                     config.CHECKPOINT_DIR, "autoencoder")
                wandb.log({"samples": wandb.Image(str(img_path))})
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"autoencoder_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

def train_ddpm(config):
    """Training loop for DDPM model"""
    device = torch.device(config.DEVICE)
    
    # Create model
    unet_config = {
        "in_channels": 1,
        "out_channels": 1,
        "features": config.UNET_CHANNELS,
        "attention_layers": config.UNET_ATTENTION_LAYERS,
    }
    
    model = DDPM(
        unet_config=unet_config,
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize wandb
    wandb.init(project=config.WANDB_PROJECT, name="ddpm")
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}") as pbar:
            for batch in pbar:
                clean = batch["clean"].to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    loss = model(clean)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                val_loss += model(clean).item()
        
        val_loss /= len(val_loader)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        # Generate and save samples
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Get a batch of real images
                sample_batch = next(iter(val_loader))
                clean = sample_batch["clean"].to(device)
                noisy = sample_batch["noisy"].to(device)
                
                # Generate denoised images
                shape = (clean.shape[0], 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
                denoised = model.sample(shape)
                
                img_path = save_images(clean, noisy, denoised, epoch + 1, 
                                     config.CHECKPOINT_DIR, "ddpm")
                wandb.log({"samples": wandb.Image(str(img_path))})
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"ddpm_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    psnr_values = []
    ssim_values = []
    
    # Create directories for saving results
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Save sample images periodically
            if batch_idx % 50 == 0:
                save_path, metrics = save_comparison_image(
                    noisy_imgs[0], clean_imgs[0], outputs[0],
                    epoch, save_dir / 'samples', batch_idx
                )
                print(f"\nBatch {batch_idx}")
                print(f"PSNR - Noisy: {metrics['noisy_psnr']:.2f}, Denoised: {metrics['denoised_psnr']:.2f}")
                print(f"SSIM - Noisy: {metrics['noisy_ssim']:.4f}, Denoised: {metrics['denoised_ssim']:.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Evaluate model
        eval_metrics = evaluate_model(model, val_loader, device)
        psnr_values.append(eval_metrics['avg_psnr'])
        ssim_values.append(eval_metrics['avg_ssim'])
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"\nNew best model saved! (Val Loss: {avg_val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'eval_metrics': eval_metrics
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"PSNR: {eval_metrics['avg_psnr']:.2f}, SSIM: {eval_metrics['avg_ssim']:.4f}")
        print("-" * 50)
        
        # Save training progress plot
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(131)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        
        # Plot PSNR
        plt.subplot(132)
        plt.plot(psnr_values, label='PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.title('Peak Signal-to-Noise Ratio')
        
        # Plot SSIM
        plt.subplot(133)
        plt.plot(ssim_values, label='SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.title('Structural Similarity Index')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_progress.png')
        plt.close()
        
        # Learning rate scheduler
        scheduler.step(avg_val_loss)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'psnr_values': psnr_values,
        'ssim_values': ssim_values
    }

def main():
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path('results')
    save_dir.mkdir(exist_ok=True)
    
    # Hyperparameters optimized for CPU
    batch_size = 4  # Smaller batch size for CPU
    num_epochs = 30  # Reduced number of epochs
    learning_rate = 0.0005  # Lower learning rate for stability
    noise_level = 0.1
    
    print("\nTraining Configuration:")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Noise Level: {noise_level}")
    
    # Create datasets and dataloaders
    train_dataset = MRIDataset('data/train', noise_level=noise_level)
    val_dataset = MRIDataset('data/val', noise_level=noise_level)
    
    # Use num_workers=0 for CPU training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset Size:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model, loss function, and optimizer
    model = LightDenoisingAutoencoder(dropout_rate=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Using Adam instead of AdamW for less memory
    
    # Learning rate scheduler with relaxed parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=7, verbose=True
    )
    
    # Initialize model weights
    model._initialize_weights()
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Train the model
    train_metrics = train_model(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, num_epochs, device, save_dir
    )
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    final_metrics = evaluate_model(model, val_loader, device)
    print("\nFinal Results:")
    print("=" * 50)
    print(f"PSNR: {final_metrics['avg_psnr']:.2f}")
    print(f"SSIM: {final_metrics['avg_ssim']:.4f}")
    print("Training complete!")

if __name__ == "__main__":
    main() 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
from models import DiffusionUNetWithAttn, VQGAN, VQGANDiscriminator
from diffusion import GaussianDiffusion
from trainer import DiffusionTrainer, VQGANTrainer
from data_loader import get_train_val_test_loaders
import torch.nn as nn

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
   
    current_dir = os.getcwd()
    
    
    data_dir = current_dir
    print(f"Data directory: {data_dir}")
    

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
  
    ct_dir = os.path.join(data_dir, 'CT')
    t1_dir = os.path.join(data_dir, 'T1')
  
    if not os.path.exists(ct_dir):
        raise ValueError(f"CT directory does not exist: {ct_dir}")
    
    if not os.path.exists(t1_dir):
        raise ValueError(f"T1 directory does not exist: {t1_dir}")
    
 
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generated_images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generated_images/test'), exist_ok=True)
    
   
    metrics_save_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_save_dir, exist_ok=True)
    
   
    print("Loading data...")
    train_loader, val_loader, test_loader, test_paths = get_train_val_test_loaders(
        data_dir, batch_size=1, val_size=0.1, test_size=0.2)

 
    print("\n=== Data range check ===")
    first_batch = next(iter(train_loader))
    mri_batch = first_batch['MRI']
    pet_batch = first_batch['PET']
    print(f"MRI range: [{mri_batch.min().item():.4f}, {mri_batch.max().item():.4f}]")
    print(f"PET range: [{pet_batch.min().item():.4f}, {pet_batch.max().item():.4f}]")
    print(f"Input image size: MRI {mri_batch.shape}, PET {pet_batch.shape}")

 
    print("\nTraining FSQ-GAN...")
    

    fsq_levels = [256, 256] 
    latent_dim = len(fsq_levels) 
    
   
    fsqgan = VQGAN(
        in_channels=1, 
        out_channels=1, 
        latent_dim=latent_dim, 
        use_fsq=True,          
        fsq_levels=fsq_levels,  
        ngf=32, 
        beta=0.25
    ).to(device)
    
    discriminator = VQGANDiscriminator(in_channels=1).to(device)
    fsqgan_save_path = os.path.join(save_dir, 'models', 'fsqgan.pth')
    
    if os.path.exists(fsqgan_save_path):
        checkpoint = torch.load(fsqgan_save_path, map_location=device)
        fsqgan.load_state_dict(checkpoint['vqgan'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        print(f"Loaded FSQ-GAN from {fsqgan_save_path}")
        print(f"FSQ configuration: levels {fsq_levels}, total codebook size {fsqgan.quantize.total_levels}")
    else:
        print("No pre-trained FSQ-GAN found, training from scratch...")
        print(f"FSQ configuration: levels {fsq_levels}, total codebook size {fsqgan.quantize.total_levels}")
        
        fsqgan_trainer = VQGANTrainer(
            fsqgan, discriminator, device, 
            lr=2e-4, recon_weight=1.0, adv_weight=0.25, vq_weight=1.0,
            patience=10, min_delta=0.001, use_amp=True
        )
        fsqgan_trainer.train_vqgan(train_loader, val_loader=val_loader, epochs=300, save_path=fsqgan_save_path)
    
    fsqgan.eval()  

 
    unet = DiffusionUNetWithAttn(
        input_nc=latent_dim,     
        context_dim=latent_dim,
        cond_dim=96,           
        output_nc=latent_dim,   
        ngf=96,                
        time_emb_dim=128,       
        norm_groups=32,          
        num_timesteps=1000, 
        res_blocks=1,           
        attn_res=(5,6,),          
        n_heads=1               
    ).to(device)
    

    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    unet.apply(init_weights)
    
    unet_save_path = os.path.join(save_dir, 'models', 'best_fsq_ldm.pth')
    
    if os.path.exists(unet_save_path):
        checkpoint = torch.load(unet_save_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            current_state_dict = unet.state_dict()
            
        
            matched_state_dict = {}
            for k, v in model_state_dict.items():
                if k in current_state_dict and v.shape == current_state_dict[k].shape:
                    matched_state_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch")
            
            unet.load_state_dict(matched_state_dict, strict=False)
            print(f"Loaded UNet from {unet_save_path} (partial weights loaded)")
        else:
            print("Checkpoint format not recognized, initializing from scratch")
    else:
        print("No pre-trained UNet found, initializing from scratch")

  
    diffusion = GaussianDiffusion(unet, fsqgan, device, num_timesteps=1000, ddim_timesteps=50, schedule_type="linear")
    

    print("\nTraining FSQ-based Latent Diffusion Model...")
    model_save_path = os.path.join(save_dir, 'models', 'best_fsq_ldm.pth')
    trainer = DiffusionTrainer(
        diffusion, device, 
        lr=1e-4,
        beta1=0.5, 
        beta2=0.9, 
        use_adamw=True,
        weight_decay=0,
        num_epochs=4000,
        scheduler_type='cosine',
        ema_decay=0.9995,
        accumulation_steps=8,
        min_lr=1e-6,
        step_start_ema=200,
        use_amp=True
    )
    
    trainer.train(
        train_loader, 
        val_loader, 
        test_loader=test_loader,
        test_paths=test_paths,
        num_epochs=4000, 
        model_save_path=model_save_path,
        metrics_save_dir=metrics_save_dir
    )
    

    print("\nEvaluating on test set...")
    test_save_dir = os.path.join(save_dir, 'generated_images/test')
    test_metrics_name = 'test_fsq_metrics.csv'
    test_mse, test_nrmse, test_mae, test_ssim, test_psnr, test_metrics = trainer.test(
        test_loader, test_paths, test_save_dir, metrics_save_dir, test_metrics_name
    )
    print(f'Test Results - MSE: {test_mse:.4f}, NRMSE: {test_nrmse:.4f}, MAE: {test_mae:.4f}, '
          f'SSIM: {test_ssim:.4f}, PSNR: {test_psnr:.4f}')
    
   
    print(f"\nFSQ configuration information:")
    print(f"  FSQ级别: {fsq_levels}")
    print(f"  Total codebook size: {fsqgan.quantize.total_levels}")
    print(f"  Latent dimension: {latent_dim}")
    
    print("\nFSQ-GAN training and evaluation completed!")

if __name__ == "__main__":
    main()
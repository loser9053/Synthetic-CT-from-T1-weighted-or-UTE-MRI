import os
import time
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import copy


try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
    print("Mixed precision training (AMP) available")
except ImportError:
    AMP_AVAILABLE = False
    print("Mixed precision training (AMP) not available, using normal training")


class VQGANTrainer:
    def __init__(self, vqgan, discriminator, device, lr=1e-5, recon_weight=5.0, 
                 adv_weight=0.25, vq_weight=1.0, gp_weight=20, n_critic=3,
                 patience=20, min_delta=0.001, use_amp=True, fft_weight=0.1):  
        self.vqgan = vqgan.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
      
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
      
        self.use_amp = use_amp and AMP_AVAILABLE and device.type == 'cuda'
        if self.use_amp:
            self.scaler_g = GradScaler()
            self.scaler_d = GradScaler()
            print("VQ-GAN trainer: enabling mixed precision training")
        else:
            self.scaler_g = None
            self.scaler_d = None
            print("VQ-GAN   训练器: 使用普通精度训练")
        
   
        self.optimizer_g = optim.Adam(
            list(self.vqgan.encoder.parameters()) + 
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.quantize.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        
        self.optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        
        self.recon_weight = recon_weight
        self.adv_weight = adv_weight
        self.vq_weight = vq_weight
        self.gp_weight = gp_weight
        self.n_critic = n_critic
        self.fft_weight = fft_weight  
        
        print(f"VQ-GAN trainer: FFT loss weight = {fft_weight}")

    def fft_loss(self, pred, target):
     
        pred_f32 = pred.float()
        target_f32 = target.float()
        pred_fft = torch.fft.rfftn(pred_f32, dim=(-3, -2, -1))
        target_fft = torch.fft.rfftn(target_f32, dim=(-3, -2, -1))
        return F.l1_loss(pred_fft, target_fft)
    
    def check_early_stop(self, current_loss):
        
        if current_loss < self.best_loss - self.min_delta:
          
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
            return self.early_stop
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
       
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        grad_outputs = torch.ones_like(d_interpolates, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_vqgan(self, train_loader, val_loader=None, epochs=100, save_path='vqgan.pth'):
        self.vqgan.train()
        self.discriminator.train()
        best_loss = float('inf')
        
      
        train_history = {
            'recon_loss': [],
            'vq_loss': [],
            'disc_loss': [],
            'perplexity': [],
            'fft_loss': [], 
            'val_loss': [] if val_loader else None
        }
        
        for epoch in range(epochs):
            if self.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
                
            print(f"VQ-GAN Epoch {epoch+1}/{epochs}:")
            
            recon_loss_epoch = 0.0
            vq_loss_epoch = 0.0
            disc_loss_epoch = 0.0
            perplexity_epoch = 0.0
            fft_loss_epoch = 0.0  
            length = len(train_loader.dataset)
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training VQ-GAN')):
                pet = batch['PET'].to(self.device)
                
                
                for _ in range(self.n_critic):
                    if self.use_amp:
                        with autocast():
                            real_logits = self.discriminator(pet)
                            with torch.no_grad():
                                recon_pet, vq_loss, encoding_inds, perplexity = self.vqgan(pet)
                            fake_logits = self.discriminator(recon_pet.detach())
                            gradient_penalty = self.compute_gradient_penalty(pet, recon_pet.detach())
                            d_loss = -torch.mean(real_logits) + torch.mean(fake_logits) + self.gp_weight * gradient_penalty
                        
                        self.optimizer_d.zero_grad()
                        self.scaler_d.scale(d_loss).backward()
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                    else:
                        self.optimizer_d.zero_grad()
                        real_logits = self.discriminator(pet)
                        with torch.no_grad():
                            recon_pet, vq_loss, encoding_inds, perplexity = self.vqgan(pet)
                        fake_logits = self.discriminator(recon_pet.detach())
                        gradient_penalty = self.compute_gradient_penalty(pet, recon_pet.detach())
                        d_loss = -torch.mean(real_logits) + torch.mean(fake_logits) + self.gp_weight * gradient_penalty
                        d_loss.backward()
                        self.optimizer_d.step()
                
                
                if self.use_amp:
                    with autocast():
                        recon_pet, vq_loss, encoding_inds, perplexity = self.vqgan(pet)
                        perplexity_epoch += perplexity.item()
                        
                        recon_loss = F.l1_loss(pet, recon_pet)
                        fft_loss = self.fft_loss(recon_pet, pet)  
                        
                        fake_logits = self.discriminator(recon_pet)
                        g_loss = -torch.mean(fake_logits)
                        
                       
                        loss = recon_loss * self.recon_weight + \
                            vq_loss * self.vq_weight + \
                            g_loss * self.adv_weight + \
                            fft_loss * self.fft_weight 
                    
                    self.optimizer_g.zero_grad()
                    self.scaler_g.scale(loss).backward()
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                else:
                    self.optimizer_g.zero_grad()
                    
                    recon_pet, vq_loss, encoding_inds, perplexity = self.vqgan(pet)
                    perplexity_epoch += perplexity.item()
                    
                    recon_loss = F.l1_loss(pet, recon_pet)
                    fft_loss = self.fft_loss(recon_pet, pet)  
                    
                    fake_logits = self.discriminator(recon_pet)
                    g_loss = -torch.mean(fake_logits)
                    
                    
                    loss = recon_loss * self.recon_weight + \
                        vq_loss * self.vq_weight + \
                        g_loss * self.adv_weight + \
                        fft_loss * self.fft_weight  
                    
                    loss.backward()
                    self.optimizer_g.step()
                
                recon_loss_epoch += recon_loss.item() * pet.size(0)
                vq_loss_epoch += vq_loss.item() * pet.size(0)
                disc_loss_epoch += d_loss.item() * pet.size(0)
                fft_loss_epoch += fft_loss.item() * pet.size(0)  
            
           
            avg_recon_loss = recon_loss_epoch / length
            avg_vq_loss = vq_loss_epoch / length
            avg_disc_loss = disc_loss_epoch / length
            avg_perplexity = perplexity_epoch / len(train_loader)
            avg_fft_loss = fft_loss_epoch / length 
            total_loss = avg_recon_loss + avg_vq_loss + avg_fft_loss * self.fft_weight
            
          
            train_history['recon_loss'].append(avg_recon_loss)
            train_history['vq_loss'].append(avg_vq_loss)
            train_history['disc_loss'].append(avg_disc_loss)
            train_history['perplexity'].append(avg_perplexity)
            train_history['fft_loss'].append(avg_fft_loss)  
            
            
            if val_loader:
                val_loss = self.validate(val_loader)
                train_history['val_loss'].append(val_loss)
                print(f"  Val Loss: {val_loss:.4f}")
                should_stop = self.check_early_stop(val_loss)
            else:
                should_stop = self.check_early_stop(total_loss)
            
            print(f"  Recon Loss: {avg_recon_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, FFT Loss: {avg_fft_loss:.4f}")
            
           
            if (epoch + 1) % 10 == 0:
                print(f"  Perplexity: {avg_perplexity:.2f} (codebook usage rate)")
            
          
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save({
                    'vqgan': self.vqgan.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'train_history': train_history
                }, save_path)
                print(f"  Best model saved with total loss: {best_loss:.4f}")
            
            # 定期检查输出
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    test_recon, _, _, _ = self.vqgan(pet[:1])
                    print(f"  VQ-GAN output range: [{test_recon.min().item():.4f}, {test_recon.max().item():.4f}]")
                    
                    # 检查码本使用情况
                    z = self.vqgan.encoder(pet[:5])
                    z_q, vq_loss, encoding_inds, _ = self.vqgan.quantize(z)
                    unique_codes = len(torch.unique(encoding_inds))
                    print(f"  Used codebook vectors: {unique_codes}/{self.vqgan.quantize.K}")
        
        return train_history
    
    
    def validate(self, val_loader):
    
        self.vqgan.eval()
        self.discriminator.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_fft_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pet = batch['PET'].to(self.device)
                batch_size = pet.size(0)
                
               
                if self.use_amp:
                    with autocast():
                        recon_pet, vq_loss, _, _ = self.vqgan(pet)
                        recon_loss = F.l1_loss(pet, recon_pet)
                        fft_loss = self.fft_loss(recon_pet, pet)
                        loss = recon_loss * self.recon_weight + vq_loss * self.vq_weight + fft_loss * self.fft_weight
                else:
                    recon_pet, vq_loss, _, _ = self.vqgan(pet)
                    recon_loss = F.l1_loss(pet, recon_pet)
                    fft_loss = self.fft_loss(recon_pet, pet)
                    loss = recon_loss * self.recon_weight + vq_loss * self.vq_weight + fft_loss * self.fft_weight
                
                total_loss += loss.item() * batch_size
                total_recon_loss += recon_loss.item() * batch_size
                total_vq_loss += vq_loss.item() * batch_size
                total_fft_loss += fft_loss.item() * batch_size
                total_samples += batch_size
        
        
        self.vqgan.train()
        self.discriminator.train()
        
        avg_loss = total_loss / total_samples
        avg_recon = total_recon_loss / total_samples
        avg_vq = total_vq_loss / total_samples
        avg_fft = total_fft_loss / total_samples
        
        print(f"  Val detailed loss - Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, FFT: {avg_fft:.4f}")
        
        return avg_loss


class DiffusionTrainer:
    def __init__(self, diffusion_model, device, lr=0.0001, beta1=0.9, beta2=0.999, 
                 use_adamw=True, weight_decay=1e-4, num_epochs=100, scheduler_type='cosine',
                 ema_decay=0.995, accumulation_steps=4, 
                 min_lr=1e-6, step_start_ema=2000, use_amp=True):  
        self.device = device
        self.diffusion = diffusion_model
        self.model = self.diffusion.model
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.min_lr = min_lr
        self.base_lr = lr
        
  
        self.use_amp = use_amp and AMP_AVAILABLE and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print("Diffusion model trainer: enabling mixed precision training")
        else:
            self.scaler = None
            print("Diffusion model trainer: using normal precision training")
        
     
        self.ema_decay = ema_decay
        self.step_start_ema = step_start_ema
        self.global_step = 0  
        
      
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()
        
      
        if use_adamw:
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                betas=(beta1, beta2),
                weight_decay=weight_decay,
                eps=1e-8
            )
            print(f"Using AdamW optimizer, weight_decay={weight_decay}")
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                betas=(beta1, beta2)
            )
            print("Using Adam optimizer")
        
    
        if scheduler_type == 'cosine':
            self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=min_lr
            )
            self.exp_lr_scheduler = None
            print(f"Using cosine annealing scheduler, T_max={num_epochs}, eta_min={min_lr}")
        elif scheduler_type == 'step':
            self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
            print("Using StepLR scheduler, step_size=20, gamma=0.5")
        elif scheduler_type == 'plateau':
            self.exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
            print("Using ReduceLROnPlateau scheduler")
        else:
            self.exp_lr_scheduler = None
            print("Not using learning rate scheduler")
        
        self.scheduler_type = scheduler_type
        print(f"Using EMA training, decay coefficient={ema_decay}, starting from step {step_start_ema}")
    
    def adjust_learning_rate(self, epoch):
        
        if self.scheduler_type == 'cosine':
           
            self.cosine_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
        elif self.scheduler_type == 'step' and self.exp_lr_scheduler is not None:
         
            self.exp_lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
        else:
            
            lr = self.optimizer.param_groups[0]['lr']
        
        return lr
    
    def save_lr_history(self, lr_history, save_dir):
        
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'learning_rate_history.csv')
        
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Learning Rate'])
            for epoch, lr in enumerate(lr_history):
                writer.writerow([epoch, lr])
        
        print(f"Learning rate history saved to {file_path}")
        
       
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(lr_history)), lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'learning_rate_curve.png'))
        plt.close()
        
    
    def update_ema(self):
       
        if self.global_step < self.step_start_ema:
            return  

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * model_param.data
    
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        epoch_count = 0
        total_grad_norm = 0.0
        grad_norm_count = 0
        
        v_vis_dir = os.path.join('./results', 'v_visualization')
        os.makedirs(v_vis_dir, exist_ok=True)
        visualize_batch_idx = np.random.randint(0, len(train_loader)) if len(train_loader) > 0 else -1

       
        accumulation_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
            mri = batch['MRI'].to(self.device)
            pet = batch['PET'].to(self.device)
            batch_size = mri.size(0)
            epoch_count += batch_size
            
            
            if accumulation_count == 0:
                self.optimizer.zero_grad()
            
            get_features = (batch_idx == visualize_batch_idx)
            
            
            if self.use_amp:
                with autocast():
                    loss, v_target, pred_v, t_batch = self.diffusion(pet, mri)
            else:
                loss, v_target, pred_v, t_batch = self.diffusion(pet, mri)
            
          
            loss = loss / self.accumulation_steps
            
            if batch_idx == visualize_batch_idx:
                current_epoch = self.current_epoch
                self.visualize_v_comparison(
                    v_target=v_target,
                    pred_v=pred_v,
                    epoch=current_epoch,
                    batch_idx=batch_idx,
                    save_dir=v_vis_dir
                )
            
            if batch_idx % 50 == 0:
                unique_t, counts = torch.unique(t_batch, return_counts=True)
                t_stats = {f"t_{t.item()}": count.item() for t, count in zip(unique_t, counts)}
                
              
                time_ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
                range_stats = {}
                
                for t_min, t_max in time_ranges:
                    mask = (t_batch >= t_min) & (t_batch < t_max)
                    if mask.any():
                        range_name = f"t_{t_min}-{t_max-1}"
                        range_stats[range_name] = mask.sum().item()
                        
                       
                        t_v_target = v_target[mask]
                        t_pred_v = pred_v[mask]
                        range_stats[f"{range_name}_v_target_std"] = t_v_target.std().item()
                        range_stats[f"{range_name}_pred_v_std"] = t_pred_v.std().item()
                        range_stats[f"{range_name}_mse"] = F.mse_loss(t_pred_v, t_v_target).item()
                
              
                print(f"  Time step distribution: {t_stats}")
                print(f"  Range stats: {range_stats}")
            
           
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_count += 1
            
          
            if accumulation_count == self.accumulation_steps:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    total_grad_norm += grad_norm
                    grad_norm_count += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}: Gradient Norm = {grad_norm:.6f}")
                    
                  
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    total_grad_norm += grad_norm
                    grad_norm_count += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}: Gradient Norm = {grad_norm:.6f}")
                    
                  
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    
                    self.optimizer.step()
                
               
                self.global_step += 1
                
                
                if self.global_step >= self.step_start_ema:
                    self.update_ema()
                    if batch_idx % 50 == 0:  
                        print(f"  EMA updated: global step {self.global_step} (starting step {self.step_start_ema})")
                else:
                    if batch_idx % 50 == 0:
                        print(f"  EMA not enabled: global step {self.global_step} (starting step {self.step_start_ema})")
              
                
           
                accumulation_count = 0
            
            epoch_loss += loss.item() * batch_size * self.accumulation_steps  
        
        
        if accumulation_count > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                
               
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5
                total_grad_norm += grad_norm
                grad_norm_count += 1
                
                print(f"Final batch: Gradient Norm = {grad_norm:.6f}")
                
               
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
               
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5
                total_grad_norm += grad_norm
                grad_norm_count += 1
                
                print(f"Final batch: Gradient Norm = {grad_norm:.6f}")
                
               
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.optimizer.step()
            
            
            self.global_step += 1
            if self.global_step >= self.step_start_ema:
                self.update_ema()
                print(f"  Final batch EMA updated: global step {self.global_step}")
            # ============================================
        
        if grad_norm_count > 0:
            avg_grad_norm = total_grad_norm / grad_norm_count
            print(f"Epoch {self.current_epoch}: average gradient norm = {avg_grad_norm:.6f}")
        
        if self.global_step >= self.step_start_ema:
            print(f"Epoch {self.current_epoch} ended: EMA enabled (global step: {self.global_step})")
        else:
            print(f"Epoch {self.current_epoch} ended: EMA not    enabled (global step: {self.global_step}, needs to reach {self.step_start_ema})")
        
        avg_loss = epoch_loss / epoch_count
        return avg_loss

    def visualize_v_comparison(self, v_target, pred_v, epoch, batch_idx, save_dir):
       
        os.makedirs(save_dir, exist_ok=True)
        
        
        v_target_sample = v_target[0].cpu().detach()  # [C, D, H, W]
        pred_v_sample = pred_v[0].cpu().detach()
        
        
        mid_depth = v_target_sample.shape[1] // 2
        v_target_slice = v_target_sample[0, mid_depth]  # [H, W]
        pred_v_slice = pred_v_sample[0, mid_depth]
        
      
        def normalize(x):
            x_min = x.min()
            x_max = x.max()
            return (x - x_min) / (x_max - x_min + 1e-8)
        
        v_target_slice_norm = normalize(v_target_slice)
        pred_v_slice_norm = normalize(pred_v_slice)
        
       
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(v_target_slice_norm, cmap='gray')
        axes[0].set_title('True v')
        axes[0].axis('off')
        
        axes[1].imshow(pred_v_slice_norm, cmap='gray')
        axes[1].set_title('Predicted v')
        axes[1].axis('off')
        
      
        save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}_v_comparison.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"v comparison saved to {save_path}")

    
    def generate_and_evaluate(self, data_loader, is_train=False, full_eval=False, ablation=False):
        self.model.eval()
        metrics = []
        printed_output = False
        with torch.no_grad():
            if full_eval:
                batches = tqdm(data_loader, desc="Evaluating all samples")
            else:
                batches = [next(iter(data_loader))]
            
            for batch in batches:
                mri = batch['MRI'].to(self.device)
                pet = batch['PET'].to(self.device)
                filenames = batch['filename']
                
               
                min_vals = batch['pet_min_pop'].to(self.device)
                max_vals = batch['pet_p995_pop'].to(self.device)
                
                if ablation:
                    fake_pet = self.diffusion.ddim_sample_loop_zero_condition_angular(mri)
                    print("Using zero condition (ablation study)")
                else:
                    fake_pet = self.diffusion.ddim_sample_loop_with_cfg_angular_optimized(mri, cfg_scale=3.5)
                
               
                fake_pet_denorm = self.denormalize_ct(fake_pet, min_vals, max_vals)
                
                
                pet_denorm = self.denormalize_ct(pet, min_vals, max_vals)
                
                
                real_img = pet_denorm.cpu().numpy()
                fake_img = fake_pet_denorm.cpu().numpy()
                
             
                if not printed_output:
                    print(f"\nGenerator output range: [{fake_pet_denorm.min().item():.4f}, {fake_pet_denorm.max().item():.4f}]")
                    print(f"Real CT range: [{pet_denorm.min().item():.4f}, {pet_denorm.max().item():.4f}]")
                    printed_output = True
                
                
                for i in range(real_img.shape[0]):
                    real_slice = real_img[i, 0]
                    fake_slice = fake_img[i, 0]
                    data_range = real_slice.max() - real_slice.min()
                    if data_range < 1e-8:
                        data_range = 1e-8
                    
                    mse = np.mean((real_slice - fake_slice) **2)
                    rmse = np.sqrt(mse)
                    nrmse = rmse / data_range
                    mae = np.mean(np.abs(real_slice - fake_slice))
                    ssim_score = ssim(real_slice, fake_slice, data_range=data_range)
                    psnr_score = psnr(real_slice, fake_slice, data_range=data_range)
                    
                    metrics.append({
                        'filename': filenames[i],
                        'epoch': self.current_epoch,
                        'MSE': mse,
                        'NRMSE': nrmse,
                        'MAE': mae,
                        'SSIM': ssim_score,
                        'PSNR': psnr_score
                    })
        
        self.model.train()
        return metrics

    def denormalize_ct(self, tensor, min_val, max_val):
        
        return tensor * (max_val - min_val) + min_val

    def save_metrics_to_csv(self, metrics_list, file_path):
       
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not metrics_list:
            print("No metrics to save.")
            return
        fieldnames = metrics_list[0].keys()
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in metrics_list:
                writer.writerow(metrics)
        print(f"Metrics saved to {file_path}")
    
    
    def validate(self, val_loader, full_metrics=False, use_cfg=False, cfg_scale=3.5):
        self.ema_model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_nrmse = 0.0
        total_mae = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        img_count = 0
        val_image_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                mri = batch['MRI'].to(self.device)
                pet = batch['PET'].to(self.device)
                filenames = batch['filename']
                batch_size = mri.size(0)
                img_count += batch_size
                
            
                min_vals = batch['pet_min_pop'].to(self.device)
                max_vals = batch['pet_p995_pop'].to(self.device)
                
               
                loss, _, _, _ = self.diffusion(pet, mri)
                
                if full_metrics:
                    original_model = self.diffusion.model
                    self.diffusion.model = self.ema_model
                    if use_cfg:
                        fake_pet = self.diffusion.ddim_sample_loop_with_cfg_angular_optimized(mri, cfg_scale=cfg_scale)
                    else:
                        fake_pet = self.diffusion.ddim_sample_loop_with_cfg_angular_optimized(mri, cfg_scale=1.0)
                    self.diffusion.model = original_model
                    
                
                    fake_pet_denorm = self.denormalize_ct(fake_pet, min_vals, max_vals)
                    pet_denorm = self.denormalize_ct(pet, min_vals, max_vals)
                    
                    real_img = pet_denorm.cpu().numpy()
                    fake_img = fake_pet_denorm.cpu().numpy()
                    
                    for i in range(batch_size):
                        real_slice = real_img[i, 0]
                        fake_slice = fake_img[i, 0]
                        data_range = real_slice.max() - real_slice.min()
                        if data_range < 1e-8:
                            data_range = 1e-8
                        
                        mse = np.mean((real_slice - fake_slice) **2)
                        rmse = np.sqrt(mse)
                        nrmse = rmse / data_range
                        mae = np.mean(np.abs(real_slice - fake_slice))
                        ssim_score = ssim(real_slice, fake_slice, data_range=data_range)
                        psnr_score = psnr(real_slice, fake_slice, data_range=data_range)
                        
                        total_mse += mse
                        total_nrmse += nrmse
                        total_mae += mae
                        total_ssim += ssim_score
                        total_psnr += psnr_score
                        
                        val_image_metrics.append({
                            'filename': filenames[i],
                            'MSE': mse,
                            'NRMSE': nrmse,
                            'MAE': mae,
                            'SSIM': ssim_score,
                            'PSNR': psnr_score
                        })
                
                total_loss += loss.item() * batch_size
        
        self.model.train()
        
        avg_loss = total_loss / img_count
        
        if full_metrics:
            avg_mse = total_mse / img_count
            avg_nrmse = total_nrmse / img_count
            avg_mae = total_mae / img_count
            avg_ssim = total_ssim / img_count
            avg_psnr = total_psnr / img_count
            return avg_loss, avg_mse, avg_nrmse, avg_mae, avg_ssim, avg_psnr, val_image_metrics
        else:
            return avg_loss, 0, 0, 0, 0, 0, []

    def test(self, test_loader, test_paths, save_dir, metrics_save_dir, metrics_save_name, cfg_scale=3.5):
        self.ema_model.eval()
        
        total_mse = 0.0
        total_nrmse = 0.0
        total_mae = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        img_count = 0
        test_image_metrics = []
        
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc='Testing')):
                mri = batch['MRI'].to(self.device)
                pet = batch['PET'].to(self.device)
                filenames = batch['filename']
                
           
                min_vals = batch['pet_min_pop'].to(self.device)
                max_vals = batch['pet_p995_pop'].to(self.device)
                
                original_model = self.diffusion.model
                self.diffusion.model = self.ema_model
                fake_pet = self.diffusion.ddim_sample_loop_with_cfg_angular_optimized(mri, cfg_scale=cfg_scale)
                self.diffusion.model = original_model
                
              
                fake_pet_denorm = self.denormalize_ct(fake_pet, min_vals, max_vals)
                
              
                pet_denorm = self.denormalize_ct(pet, min_vals, max_vals)
                
                fake_pet_numpy = fake_pet_denorm.cpu().numpy()
                pet_numpy = pet_denorm.cpu().numpy()
                
                for j in range(fake_pet_numpy.shape[0]):
                  
                    fake_pet_save = fake_pet_numpy[j, 0].astype(np.float32)
                    _, pet_path = test_paths[i * test_loader.batch_size + j]
                    filename = os.path.basename(pet_path)
                    save_path = os.path.join(save_dir, f'generated_{filename}')
                    
                    original_img = nib.load(pet_path)
                    affine = original_img.affine
                    generated_img = nib.Nifti1Image(fake_pet_save, affine)
                    nib.save(generated_img, save_path)
                    
                    real_img = pet_numpy[j, 0].astype(np.float32)
                    fake_img = fake_pet_numpy[j, 0].astype(np.float32)
                    data_range = max(real_img.max() - real_img.min(), 1e-8)
                    
                    mse = np.mean((real_img - fake_img)** 2)
                    rmse = np.sqrt(mse)
                    nrmse = rmse / data_range
                    mae = np.mean(np.abs(real_img - fake_img))
                    ssim_score = ssim(real_img, fake_img, data_range=data_range)
                    psnr_score = psnr(real_img, fake_img, data_range=data_range)
                    
                    total_mse += mse
                    total_nrmse += nrmse
                    total_mae += mae
                    total_ssim += ssim_score
                    total_psnr += psnr_score
                    
                    test_image_metrics.append({
                        'filename': filename,
                        'MSE': mse,
                        'NRMSE': nrmse,
                        'MAE': mae,
                        'SSIM': ssim_score,
                        'PSNR': psnr_score
                    })
                    
                    img_count += 1
        
        avg_mse = total_mse / img_count
        avg_nrmse = total_nrmse / img_count
        avg_mae = total_mae / img_count
        avg_ssim = total_ssim / img_count
        avg_psnr = total_psnr / img_count
        
        self.save_metrics_to_csv(test_image_metrics, os.path.join(metrics_save_dir, metrics_save_name))
        
        return avg_mse, avg_nrmse, avg_mae, avg_ssim, avg_psnr, test_image_metrics
    
    def train(self, train_loader, val_loader, test_loader, test_paths,
          num_epochs, model_save_path, metrics_save_dir):
        best_val_loss = float('inf')
        best_combined_metric = 0.0  
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_ema_model_wts = copy.deepcopy(self.ema_model.state_dict())
        all_train_metrics = []
        all_val_metrics = []
        
    
        lr_history = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            since = time.time()
            
         
            train_metrics = []
            
            if epoch == num_epochs - 1:
             
                train_loss = self.train_epoch(train_loader)
                train_metrics = self.generate_and_evaluate(train_loader, is_train=True, full_eval=True)
                print(f"Last epoch: evaluation completed for all {len(train_metrics)} samples in the training set")
            else:
            
                train_loss = self.train_epoch(train_loader)
            
          
            if train_metrics:
                all_train_metrics.extend(train_metrics)
            
           
            current_lr = self.adjust_learning_rate(epoch)
            lr_history.append(current_lr)
            
          
            full_metrics = False
            if epoch < 300:
               
                val_loss, val_mse, val_nrmse, val_mae, val_ssim, val_psnr, val_metrics = self.validate(val_loader, full_metrics=False)
            else:
               
                if (epoch + 1) % 2 == 0:
                    full_metrics = True
                    val_loss, val_mse, val_nrmse, val_mae, val_ssim, val_psnr, val_metrics = self.validate(val_loader, full_metrics=True)
                    all_val_metrics.extend(val_metrics)
                    
                    
                    combined_metric = val_psnr + 30 * val_ssim
                    
                  
                    if combined_metric > best_combined_metric:
                        best_combined_metric = combined_metric
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_ema_model_wts = copy.deepcopy(self.ema_model.state_dict())
                        torch.save({
                            'model': best_model_wts,
                            'ema_model': best_ema_model_wts,
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': val_loss,
                            'psnr': val_psnr,
                            'ssim': val_ssim,
                            'combined_metric': combined_metric
                        }, model_save_path)
                        print(f'Best PSNR+30SSIM model saved at epoch {epoch+1}: PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}, Combined={combined_metric:.4f}')
                else:
                    
                    val_loss, val_mse, val_nrmse, val_mae, val_ssim, val_psnr, val_metrics = self.validate(val_loader, full_metrics=False)
            
     
            if self.scheduler_type == 'plateau':
                self.exp_lr_scheduler.step(val_loss)
            
            time_elapsed = time.time() - since
            
        
            print(f'Epoch {epoch+1}/{num_epochs} | Time: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
            print(f'Learning Rate: {current_lr:.2e}')
            print(f'Train Loss: {train_loss:.4f}')
            
            if full_metrics:
                print(f'Val Loss: {val_loss:.4f} | '
                    f'MSE: {val_mse:.4f}, NRMSE: {val_nrmse:.4f}, MAE: {val_mae:.4f}, '
                    f'SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}')
                print(f'Combined Metric (PSNR + 30*SSIM): {combined_metric:.4f}')
            else:
                print(f'Val Loss: {val_loss:.4f}')
            
          
            if epoch < 400 and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_ema_model_wts = copy.deepcopy(self.ema_model.state_dict())
                torch.save({
                    'model': best_model_wts,
                    'ema_model': best_ema_model_wts,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss
                }, model_save_path.replace('.pth', '_best_loss.pth'))  
                print(f'Best val loss model saved at epoch {epoch+1}')
        
       
        self.save_lr_history(lr_history, metrics_save_dir)
        
       
        if best_combined_metric > 0:  
            checkpoint = torch.load(model_save_path)
            best_model_wts = checkpoint['model']
            best_ema_model_wts = checkpoint['ema_model']
            print(f'Loaded best PSNR+30SSIM model from epoch {checkpoint["epoch"]+1}')
            print(f'Best metrics: PSNR={checkpoint["psnr"]:.4f}, SSIM={checkpoint["ssim"]:.4f}, Combined={checkpoint["combined_metric"]:.4f}')
        else:
           
            checkpoint = torch.load(model_save_path.replace('.pth', '_best_loss.pth'))
            best_model_wts = checkpoint['model']
            best_ema_model_wts = checkpoint['ema_model']
            print(f'Loaded best val loss model from epoch {checkpoint["epoch"]+1}')
        
        self.model.load_state_dict(best_model_wts)
        self.ema_model.load_state_dict(best_ema_model_wts)
        
        self.save_metrics_to_csv(all_train_metrics, os.path.join(metrics_save_dir, 'train_metrics.csv'))
        self.save_metrics_to_csv(all_val_metrics, os.path.join(metrics_save_dir, 'val_metrics.csv'))
        
        print(f'Best Validation Loss: {best_val_loss:.4f}')
        if best_combined_metric > 0:
            print(f'Best Combined Metric (PSNR + 30*SSIM): {best_combined_metric:.4f}')
        print(f'All metrics saved to {metrics_save_dir}')
        
        return self.model
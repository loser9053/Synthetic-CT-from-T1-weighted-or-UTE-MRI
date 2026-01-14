import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, model, vqgan, device,
                 num_timesteps=1000, ddim_timesteps=200,
                 ddim_eta=0.5, schedule_type="cosine", latent_stats=None):
        super().__init__()
        self.model = model.to(device)
        self.vqgan = vqgan.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        
  
        self.latent_dim = self.vqgan.latent_dim


        if schedule_type == "cosine":
            alpha_t, sigma_t, phi_t = self.cosine_schedule_for_v_prediction(
                num_timesteps, s=0.008, clamp_range=(1e-5, 0.999)
            )
        elif schedule_type == "linear":
            alpha_t, sigma_t, phi_t = self.linear_schedule_for_v_prediction(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        

        self.alpha_t = alpha_t.to(device)      # √ᾱ_t
        self.sigma_t = sigma_t.to(device)      # √(1-ᾱ_t)  
        self.phi_t = phi_t.to(device)          # φ_t = atan2(σ_t, α_t)
        
  
        self.alphas_cumprod = (self.alpha_t ** 2).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = self.alpha_t
        self.sqrt_one_minus_alphas_cumprod = self.sigma_t


        self._print_angle_stats()

        # DDIM
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta
        self.ddim_timestep_seq = self._setup_ddim_timesteps()
        
  
        self._precompute_angular_tables()

    def cosine_schedule_for_v_prediction(self, timesteps: int, s: float = 0.008, 
                                       clamp_range: tuple = (1e-5, 0.999)):
       
        steps = timesteps + 1                         
        x = torch.linspace(0, timesteps, steps)
        cos_sq = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = cos_sq / cos_sq[0]            
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, *clamp_range)      

        # 重新计算，保证首尾严格
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod = torch.cat([torch.ones(1), alphas_cumprod])  # α₀=1
        sigmas = torch.sqrt(1.0 - alphas_cumprod)                    # σₜ
        alphas = torch.sqrt(alphas_cumprod)                          # αₜ
        phis = torch.atan2(sigmas, alphas)                           # φₜ


        return alphas[1:], sigmas[1:], phis[1:]

    def linear_schedule_for_v_prediction(self, timesteps: int):

        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # 重新计算确保一致性
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = torch.cat([torch.ones(1), alphas_cumprod])  # α₀=1
        
        sigmas = torch.sqrt(1.0 - alphas_cumprod)                    # σₜ
        alphas = torch.sqrt(alphas_cumprod)                          # αₜ
        phis = torch.atan2(sigmas, alphas)                           # φₜ

        # 去掉 t=0 的占位
        return alphas[1:], sigmas[1:], phis[1:]

    def _print_angle_stats(self):
  
        print("Angle Statistics:")
        print(f"  Initial angle φ₀: {self.phi_t[0].item()*180/math.pi:.2f}°")
        print(f"  Final angle φ_T: {self.phi_t[-1].item()*180/math.pi:.2f}°")
        print(f"  Angle range: {self.phi_t.min().item()*180/math.pi:.2f}° ~ {self.phi_t.max().item()*180/math.pi:.2f}°")
        print(f"  α_t range: [{self.alpha_t.min().item():.4f}, {self.alpha_t.max().item():.4f}]")
        print(f"  σ_t range: [{self.sigma_t.min().item():.4f}, {self.sigma_t.max().item():.4f}]")

    def _precompute_angular_tables(self):
  
        print("Precomputing angle and trigonometric tables...")
        
  
        ddim_seq = self.ddim_timestep_seq
        
  
        self.cos_delta_table = torch.zeros(len(ddim_seq), device=self.device)
        self.sin_delta_table = torch.zeros(len(ddim_seq), device=self.device)
        
        for i in range(len(ddim_seq)):
            if i == 0:
     
                t_current = ddim_seq[i]
                phi_current = self.phi_t[t_current - 1]  
                # φ_0 = 0
                delta = phi_current - 0.0
            else:
                t_current = ddim_seq[i]
                t_prev = ddim_seq[i-1]
                phi_current = self.phi_t[t_current - 1]  
                phi_prev = self.phi_t[t_prev - 1]        
                delta = phi_current - phi_prev
            
            self.cos_delta_table[i] = torch.cos(delta)
            self.sin_delta_table[i] = torch.sin(delta)
        
   
        self.cos_delta_table = self.cos_delta_table.view(-1, 1, 1, 1, 1)
        self.sin_delta_table = self.sin_delta_table.view(-1, 1, 1, 1, 1)
        
        print(f"Precomputation completed: cos_delta_table {self.cos_delta_table.shape}, sin_delta_table {self.sin_delta_table.shape}")

    def _setup_ddim_timesteps(self):
        if self.ddim_timesteps > self.num_timesteps:
            raise ValueError(f"DDIM timesteps ({self.ddim_timesteps}) cannot exceed total timesteps ({self.num_timesteps})")
        step_ratio = self.num_timesteps // self.ddim_timesteps
        timesteps = (torch.arange(0, self.ddim_timesteps) * step_ratio).long() + 1
        if timesteps[-1] < self.num_timesteps - 1:
            timesteps = torch.cat([timesteps, torch.tensor([self.num_timesteps - 1], dtype=torch.long)])
        return timesteps

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
    
        alpha_t = self._extract_angular(self.alpha_t, t, x0.shape)
        sigma_t = self._extract_angular(self.sigma_t, t, x0.shape)
        return alpha_t * x0 + sigma_t * noise

    def ddim_sample_loop_zero_condition_angular(self, mri):
  
        shape = (mri.shape[0], self.latent_dim, 48, 48, 40)
        z = torch.randn(shape, device=self.device)
        timesteps = self.ddim_timestep_seq
        
        print("Using angular parameterization zero condition DDIM sampling...")
        
 
        rev_timesteps = list(reversed(range(len(timesteps))))
        
        for rev_idx, i in enumerate(rev_timesteps):
            t_val = timesteps[i]
            t_prev_val = timesteps[i-1] if i > 0 else 0
            
            t = torch.full((shape[0],), t_val.item(), device=self.device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_val, device=self.device, dtype=torch.long)
            
    
            zero_context = torch.zeros_like(mri)
            z = self.ddim_sample_angular_optimized(
                z, t, t_prev, zero_context, cfg_scale=1.0, step_idx=rev_idx
            )
            
      
            if rev_idx % max(1, len(timesteps) // 5) == 0 or rev_idx == len(timesteps) - 1:
                print(f"DDIM step {rev_idx+1}/{len(timesteps)}")
        
        with torch.no_grad():
            pet_pred = self.vqgan.decode(z) 
        pet_pred = torch.clamp(pet_pred, min=0.0)
        return pet_pred
    
    def ddim_sample_angular_optimized(self, x, t, t_prev, context, cfg_scale=3.5, step_idx=None):
   
        if step_idx is None:
            raise ValueError("step_idx must be provided for optimized angular sampling")
        
  
        cos_delta = self.cos_delta_table[step_idx]
        sin_delta = self.sin_delta_table[step_idx]
        

        pred_v_cond = self.model(x, context, t)
        if cfg_scale > 1.0:
            zero_condition = torch.zeros_like(context)
            pred_v_uncond = self.model(x, zero_condition, t)
            pred_v = pred_v_uncond + cfg_scale * (pred_v_cond - pred_v_uncond)
        else:
            pred_v = pred_v_cond

   
        mean = x * cos_delta - pred_v * sin_delta
        

        if t_prev[0] == 0:
            return mean
        else:
     
            alpha_cumprod_t = self._extract_angular(self.alphas_cumprod, t, x.shape)
            alpha_cumprod_t_prev = self._extract_angular(self.alphas_cumprod, t_prev, x.shape)
            variance = self.ddim_eta ** 2 * \
                       ((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * \
                       (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise

    def ddim_sample_loop_with_cfg_angular_optimized(self, mri, cfg_scale=3.5):

        shape = (mri.shape[0], self.latent_dim, 48, 48, 40)
        z = torch.randn(shape, device=self.device)
        timesteps = self.ddim_timestep_seq
        
        print("Using optimized angular parameter for DDIM sampling...")
        
   
        rev_timesteps = list(reversed(range(len(timesteps))))
        
        for rev_idx, i in enumerate(rev_timesteps):
            t_val = timesteps[i]
            t_prev_val = timesteps[i-1] if i > 0 else 0
            
            t = torch.full((shape[0],), t_val.item(), device=self.device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_val, device=self.device, dtype=torch.long)
            
    
            z = self.ddim_sample_angular_optimized(
                z, t, t_prev, mri, cfg_scale=cfg_scale, step_idx=rev_idx
            )
            
    
            if rev_idx % max(1, len(timesteps) // 5) == 0 or rev_idx == len(timesteps) - 1:
                print(f"DDIM step {rev_idx+1}/{len(timesteps)}")
        
        with torch.no_grad():
            pet_pred = self.vqgan.decode(z)  
        pet_pred = torch.clamp(pet_pred, min=0.0)
        return pet_pred

    def _extract_angular(self, a, t, x_shape):
  
        batch_size = t.shape[0]
    
        t_index = t - 1
        t_index = torch.clamp(t_index, 0, len(a) - 1)  
        out = a[t_index]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _extract(self, a, t, x_shape):
    
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

   
    def training_losses(self, x0, mri):
        batch_size = x0.shape[0]
       
        t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=self.device).long()

        with torch.no_grad():
           
            pet_z, vq_loss = self.vqgan.encode(x0)  

        noise = torch.randn_like(pet_z)
        
        
        z_noisy = self.q_sample(x0=pet_z, t=t, noise=noise)

    
        alpha_t = self._extract_angular(self.alpha_t, t, z_noisy.shape)
        sigma_t = self._extract_angular(self.sigma_t, t, z_noisy.shape)
        v_target = alpha_t * noise - sigma_t * pet_z

   
        pred_v = self.model(z_noisy, mri, t)

   
        mse_loss = F.mse_loss(pred_v, v_target)

        return mse_loss, v_target, pred_v, t

    def forward(self, x0, mri_cond):
        return self.training_losses(x0, mri_cond)
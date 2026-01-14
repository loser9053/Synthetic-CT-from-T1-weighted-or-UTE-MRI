import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from inspect import isfunction
import numpy as np


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99, epsilon=1e-5):
       
        super(VectorQuantizerEMA, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        
   
        self.register_buffer('embedding', torch.randn(self.K, self.D))
        self.register_buffer('cluster_size', torch.zeros(self.K))
        self.register_buffer('embedding_avg', self.embedding.clone())
        
    def forward(self, latents):
    
        latents = latents.permute(0, 2, 3, 4, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        
     
        dist = torch.sum(flat_latents**2, dim=1, keepdim=True) + \
               torch.sum(self.embedding**2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.t())
        
     
        encoding_inds = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_inds, self.K).float()
        
      
        quantized = torch.matmul(encodings, self.embedding).view(latents_shape)
        
       
        if self.training:
           
            self.cluster_size = self.cluster_size * self.decay + \
                               (1 - self.decay) * torch.sum(encodings, 0)
            
          
            embed_sum = torch.matmul(encodings.t(), flat_latents)
            self.embedding_avg = self.embedding_avg * self.decay + (1 - self.decay) * embed_sum
            
          
            n = torch.sum(self.cluster_size)
            cluster_size = (
                (self.cluster_size + self.epsilon) / 
                (n + self.K * self.epsilon) * n
            )
            
           
            self.embedding = self.embedding_avg / cluster_size.unsqueeze(1)
        
     
        commitment_loss = F.mse_loss(quantized.detach(), latents)
        embedding_loss = F.mse_loss(quantized, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        
     
        quantized = latents + (quantized - latents).detach()
        
       
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized.permute(0, 4, 1, 2, 3).contiguous(), vq_loss, encoding_inds, perplexity

class FiniteScalarQuantizer(nn.Module):
    
    def __init__(self, levels, eps=1e-5):
        """
        Args:
            levels: List of integers, specifying the number of levels per dimension
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.levels = levels
        self.eps = eps
        self.num_levels = len(levels)
        self.total_levels = int(np.prod(levels))
        
      
        self.K = self.total_levels 
        
   
        self.register_buffer('quant_levels', torch.tensor(levels, dtype=torch.float32))
        
    def forward(self, z):
        """
        Args:
            z: Input tensor of shape [B, D, H, W, Depth]
        Returns:
            quantized: Quantized tensor
            commitment_loss: Commitment loss for training
            encoding_inds: Encoding indices (for compatibility)
            perplexity: Perplexity (for compatibility)
        """
        batch_size, channels, height, width, depth = z.shape
        
      
        z_reshaped = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flat = z_reshaped.view(-1, channels)
        
     
        if channels != self.num_levels:
            raise ValueError(f"Input channels ({channels}) must match number of levels ({self.num_levels})")
        
      
        quantized_flat = self._quantize(z_flat)
        
      
        quantized = quantized_flat.view(batch_size, height, width, depth, channels)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
       
        commitment_loss = F.mse_loss(quantized.detach(), z)
        
       
        encoding_inds = self._compute_encoding_inds(z_flat)
        
      
        unique_codes = len(torch.unique(encoding_inds))
        perplexity = torch.exp(-torch.sum(torch.ones(unique_codes) / unique_codes * 
                                        torch.log(torch.ones(unique_codes) / unique_codes + 1e-10)))
        
  
        quantized = z + (quantized - z).detach()
        
        return quantized, commitment_loss, encoding_inds, perplexity
    
    def _quantize(self, z):
       
        z_scaled = (z + 1.0) / 2.0  
        z_scaled = z_scaled * (self.quant_levels - 1).unsqueeze(0)
        
        
        z_rounded = torch.round(z_scaled)
        
        
        max_vals = (self.quant_levels - 1).unsqueeze(0)
     
        z_rounded = torch.minimum(torch.maximum(z_rounded, torch.tensor(0.0, device=z_rounded.device)), max_vals)
        
     
        z_quantized = z_rounded / (self.quant_levels - 1).unsqueeze(0)
        z_quantized = z_quantized * 2.0 - 1.0
        
        return z_quantized
    
    def _compute_encoding_inds(self, z):
       
        z_scaled = (z + 1.0) / 2.0
        z_scaled = z_scaled * (self.quant_levels - 1).unsqueeze(0)
        z_rounded = torch.round(z_scaled)
        
      
        max_vals = (self.quant_levels - 1).unsqueeze(0)
        z_rounded = torch.minimum(torch.maximum(z_rounded, torch.tensor(0.0, device=z_rounded.device)), max_vals)
        
      
        encoding_inds = torch.zeros(z.shape[0], device=z.device)
        for i, level in enumerate(self.levels):
            encoding_inds = encoding_inds * level + z_rounded[:, i]
        
        return encoding_inds.long()
    
    def get_num_embeddings(self):
       
        return self.total_levels

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(16, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class VQGANEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, ngf=32):
        super().__init__()
        
     
        self.down1 = nn.Sequential(
            nn.Conv3d(in_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ngf, ngf)
        )
        
      
        self.down2 = nn.Sequential(
            nn.Conv3d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ngf*2, ngf*2),
            ResidualBlock(ngf*2, ngf*2)
        )
        
   
        self.output_conv = nn.Sequential(
            nn.Conv3d(ngf*2, latent_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
       
        x = self.down1(x)  # [B, ngf, 96, 96, 80]
        x = self.down2(x)  # [B, ngf*2, 48, 48, 40]
        z = self.output_conv(x)  # [B, latent_dim, 48, 48, 40]
        
        return z  

class VQGANDecoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=4, ngf=32):
        super().__init__()

      
        self.init_conv = nn.Sequential(
            nn.Conv3d(latent_dim, ngf*2, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf*2),
            nn.ReLU(inplace=True)
        )

    
        self.res_blocks = nn.Sequential(
            ResidualBlock(ngf*2, ngf*2),
            ResidualBlock(ngf*2, ngf*2)
        )

      
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(ngf*2, ngf, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf),
            nn.ReLU(inplace=True)
        )

       
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(ngf, ngf//2, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf//2),
            nn.ReLU(inplace=True)
        )

     
        self.output_conv = nn.Sequential(
            nn.Conv3d(ngf//2, ngf//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ngf//4, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        
        x = self.init_conv(z)  # [B, ngf*2, 48, 48, 40]
        x = self.res_blocks(x)

     
        x = self.up1(x)  # [B, ngf, 96, 96, 80]

      
        x = self.up2(x)  # [B, ngf//2, 192, 192, 160]

        # 最终输出
        x = self.output_conv(x)
        return x


class VQGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, channels=[32, 32, 64, 64]):
        super().__init__()
        
        layers = []
        in_ch = in_channels
        
        for out_ch in channels:
            layers.extend([
                nn.utils.spectral_norm(nn.Conv3d(in_ch, out_ch, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_ch = out_ch
        
       
        layers.extend([
            nn.utils.spectral_norm(nn.Conv3d(channels[-1], channels[-1]*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(channels[-1]*2, 1, 4, 1, 0))
        ])
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VQGAN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=4, 
                 use_fsq=True, fsq_levels=[8, 8, 8, 8],  
                 num_embeddings=512, ngf=32, beta=0.25):
        super().__init__()
        self.encoder = VQGANEncoder(in_channels, latent_dim, ngf)
        self.decoder = VQGANDecoder(out_channels, latent_dim, ngf)
        
        if use_fsq:
            self.quantize = FiniteScalarQuantizer(levels=fsq_levels)
        else:
            self.quantize = VectorQuantizerEMA(num_embeddings, latent_dim, beta=0.25, decay=0.99)
            
        self.latent_dim = latent_dim
        self.use_fsq = use_fsq
        
    def encode(self, x):
        z = self.encoder(x)  
        z_q, vq_loss, _, _ = self.quantize(z)
        return z_q, vq_loss  
    
    def decode(self, z):
        return self.decoder(z)  
    
    def forward(self, x):
        
        z = self.encoder(x)
        z_q, vq_loss, encoding_inds, perplexity = self.quantize(z)
        
        
        recon_x = self.decoder(z_q)
        
        return recon_x, vq_loss, encoding_inds, perplexity


class MRIEncoder(nn.Module):
    def __init__(self, in_channels=3, cond_dim=256, context_dim=1, ngf=128):
        
        super().__init__()
      
        
       
        self.main = nn.Sequential(
           
            nn.Conv3d(in_channels, ngf, kernel_size=4, stride=2, padding=1),
            Swish(),
            ResidualBlock(ngf, ngf),
            
            
            nn.Conv3d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, ngf*2),
            Swish(),
            ResidualBlock(ngf*2, ngf*2),
            
          
            nn.Conv3d(ngf*2, ngf*4, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf*4),
            Swish(),
            ResidualBlock(ngf*4, ngf*4),
        )
        
       
        self.feature_proj = nn.Sequential(
            nn.Conv3d(ngf*4, ngf*2, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf*2),
            Swish(),
            nn.Conv3d(ngf*2, context_dim, kernel_size=3, padding=1),  
        )
        
       
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cond_proj = nn.Linear(ngf*4, cond_dim)
        
    def forward(self, x):
        features = self.main(x)  # [B, ngf*4, 48, 48, 40]
        
       
        feature_map = self.feature_proj(features)
        
       
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)  # [B, ngf*4]
        
      
        cond_vector = self.cond_proj(pooled)  # [B, cond_dim]
        
        return feature_map, cond_vector

# AdaGN (Adaptive Group Normalization)
class AdaGN(nn.Module):
    def __init__(self, num_channels, cond_dim, norm_groups=32, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.cond_dim = cond_dim
        self.norm_groups = norm_groups
        self.eps = eps
        
      
        self.group_norm = nn.GroupNorm(norm_groups, num_channels, eps=eps, affine=False)
        
       
        self.linear_scale = nn.Linear(cond_dim, num_channels)
        self.linear_shift = nn.Linear(cond_dim, num_channels)
        
       
        nn.init.zeros_(self.linear_scale.weight)
        nn.init.zeros_(self.linear_scale.bias)
        nn.init.zeros_(self.linear_shift.weight)
        nn.init.zeros_(self.linear_shift.bias)
        
    def forward(self, x, cond):
        

        x_norm = self.group_norm(x)
        
      
        scale = self.linear_scale(cond)  # [B, C]
        shift = self.linear_shift(cond)  # [B, C]
        
       
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)  # [B, C, 1, 1, 1]
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1, 1)  # [B, C, 1, 1, 1]
        
        
        x_ada = x_norm * (1 + scale) + shift
        
        return x_ada


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwd, bncyxz -> bnhwdyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, depth, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, depth, height, width, depth)

        out = torch.einsum("bnhwdyxz, bncyxz -> bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))

        return out + input


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, cond_dim=256, norm_groups=16, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

       
        self.block1 = Block(dim, dim_out, groups=norm_groups, dropout=dropout, cond_dim=cond_dim)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout, cond_dim=cond_dim)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, cond):
        h = self.block1(x, cond)
        if exists(self.mlp) and exists(time_emb):
            h += self.mlp(time_emb)[:, :, None, None, None]
        h = self.block2(h, cond)
        return h + self.res_conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0, cond_dim=256):
        super().__init__()
       
        self.ada_norm = AdaGN(dim, cond_dim, norm_groups=groups)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout) if dropout != 0 else nn.Identity()
        self.conv = nn.Conv3d(dim, dim_out, 3, padding=1)

    def forward(self, x, cond):
        x = self.ada_norm(x, cond)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, cond_dim=256, norm_groups=16, dropout=0, 
                 with_attn=False, num_blocks=2, n_heads=1):
        super().__init__()
        self.with_attn = with_attn
        
     
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResnetBlock(
                dim if i == 0 else dim_out, 
                dim_out, 
                time_emb_dim, 
                cond_dim=cond_dim,
                norm_groups=norm_groups, 
                dropout=dropout
            ))
        
        if with_attn:
           
            self.attn = SelfAttentionWithAdaGN(dim_out, cond_dim, n_head=n_heads, norm_groups=norm_groups)

    def forward(self, x, time_emb, cond):
        for block in self.blocks:
            x = block(x, time_emb, cond)
        
        if self.with_attn:
            x = self.attn(x, cond)
        return x


class SelfAttentionWithAdaGN(nn.Module):
    def __init__(self, in_channel, cond_dim, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.cond_dim = cond_dim

      
        self.ada_norm = AdaGN(in_channel, cond_dim, norm_groups=norm_groups)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input, cond):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

       
        norm = self.ada_norm(input, cond)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwd, bncyxz -> bnhwdyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, depth, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, depth, height, width, depth)

        out = torch.einsum("bnhwdyxz, bncyxz -> bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))

        return out + input


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim * 2  
        self.conv = nn.Conv3d(in_dim, out_dim, 4, 2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim // 2  
        self.conv = nn.ConvTranspose3d(in_dim, out_dim, 4, 2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class DiffusionUNetWithAttn(nn.Module):
    def __init__(self, input_nc=1, context_dim=1, cond_dim=128, output_nc=1, ngf=64, 
                 time_emb_dim=128, norm_groups=8, dropout=0, num_timesteps=1000, 
                 res_blocks=2, attn_res=(6,), n_heads=4):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim  
        self.context_dim = context_dim  
        self.ngf = ngf
        self.num_timesteps = num_timesteps
        self.n_heads = n_heads
        
        
        self.mri_encoder = MRIEncoder(
            in_channels=3, 
            cond_dim=cond_dim, 
            context_dim=context_dim,  
            ngf=32
        )
        
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Swish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        ) if exists(time_emb_dim) else None

        
        self.init_conv = nn.Conv3d(input_nc + context_dim, ngf, kernel_size=3, padding=1)
        
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        
        channels = [ngf, ngf*2, ngf*4, ngf*8]
        resolutions = [48, 24, 12, 6]  
        
       
        for i in range(len(channels)-1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            current_res = resolutions[i]
            use_attn = current_res in attn_res
            
            self.downs.append(ResnetBlocWithAttn(
                in_ch, in_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim,
                norm_groups=norm_groups, dropout=dropout, 
                with_attn=use_attn, num_blocks=res_blocks,
                n_heads=n_heads
            ))
            self.downs.append(Downsample(in_ch, out_ch))
        
       
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(
                channels[-1], channels[-1], time_emb_dim=time_emb_dim, cond_dim=cond_dim,
                norm_groups=norm_groups, dropout=dropout, 
                with_attn=True, num_blocks=res_blocks,
                n_heads=n_heads
            ),
            ResnetBlocWithAttn(
                channels[-1], channels[-1], time_emb_dim=time_emb_dim, cond_dim=cond_dim,
                norm_groups=norm_groups, dropout=dropout, 
                with_attn=False, num_blocks=res_blocks
            )
        ])
        
       
        for i in reversed(range(len(channels)-1)):
            out_ch = channels[i]
            current_res = resolutions[i]
            use_attn = current_res in attn_res
            
            self.ups.append(Upsample(channels[i+1], out_ch))
            self.ups.append(ResnetBlocWithAttn(
                out_ch * 2, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim,
                norm_groups=norm_groups, dropout=dropout, 
                with_attn=use_attn, num_blocks=res_blocks,
                n_heads=n_heads
            ))
        
       
        self.final_norm = AdaGN(ngf, cond_dim, norm_groups=norm_groups)
        self.final_swish = Swish()
        self.final_conv = nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1)
        
       
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mri, t):
        
        mri_feature_map, cond_vector = self.mri_encoder(mri)  # [B, context_dim, 48, 48, 40], [B, cond_dim]
        
       
        continuous_t = t.float() 
        time_emb = self.time_mlp(continuous_t) if exists(self.time_mlp) else None
        
        
        h = torch.cat([x, mri_feature_map], dim=1)  
        h = self.init_conv(h)
        
       
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, time_emb, cond_vector)
                skips.append(h)
            else:
                h = layer(h)
        
       
        for layer in self.mid:
            h = layer(h, time_emb, cond_vector)
        
       
        for layer in self.ups:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, time_emb, cond_vector)
        
        
        h = self.final_norm(h, cond_vector)
        h = self.final_swish(h)
        output = self.final_conv(h)
        
        return output
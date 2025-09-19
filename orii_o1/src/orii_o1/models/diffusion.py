"""
Orii-O1 Diffusion Model for Image Generation
A custom diffusion model with text conditioning capabilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

@dataclass
class DiffusionOutput:
    """Output structure for diffusion model."""
    sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    """Cross attention for text conditioning."""
    
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.heads
        
        if context is None:
            context = x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], h, self.dim_head).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], h, self.dim_head).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], h, self.dim_head).transpose(1, 2)
        
        # Compute attention
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        
        return self.to_out(out)

class ResnetBlock(nn.Module):
    """ResNet block with time and text conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First conv block
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Time conditioning
        time_emb = self.time_emb_proj(time_emb)
        x = x + time_emb[:, :, None, None]
        
        # Second conv block
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        # Residual connection
        return x + self.residual_conv(residual)

class AttentionBlock(nn.Module):
    """Self and cross attention block."""
    
    def __init__(self, channels: int, context_dim: Optional[int] = None, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim or channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.self_attn = CrossAttention(channels, channels, heads)
        self.cross_attn = CrossAttention(channels, self.context_dim, heads) if context_dim else None
        
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        
        # Reshape for attention
        x = self.norm(x)
        x = x.view(batch, channels, height * width).transpose(1, 2)
        
        # Self attention
        x = x + self.self_attn(x)
        
        # Cross attention (if context provided)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(x, context)
        
        # Feed forward
        x = x + self.ff(x)
        
        # Reshape back
        x = x.transpose(1, 2).view(batch, channels, height, width)
        
        return x + residual

class DownBlock(nn.Module):
    """Downsampling block with attention."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 has_attention: bool = False, context_dim: Optional[int] = None):
        super().__init__()
        self.has_attention = has_attention
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels, out_channels, time_emb_dim),
            ResnetBlock(out_channels, out_channels, time_emb_dim)
        ])
        
        if has_attention:
            self.attentions = nn.ModuleList([
                AttentionBlock(out_channels, context_dim),
                AttentionBlock(out_channels, context_dim)
            ])
        
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, time_emb)
            
            if self.has_attention:
                x = self.attentions[i](x, context)
            
            output_states = output_states + (x,)
        
        x = self.downsample(x)
        output_states = output_states + (x,)
        
        return x, output_states

class UpBlock(nn.Module):
    """Upsampling block with attention."""
    
    def __init__(self, in_channels: int, prev_output_channel: int, out_channels: int, 
                 time_emb_dim: int, has_attention: bool = False, context_dim: Optional[int] = None):
        super().__init__()
        self.has_attention = has_attention
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels + prev_output_channel, out_channels, time_emb_dim),
            ResnetBlock(out_channels + prev_output_channel, out_channels, time_emb_dim),
            ResnetBlock(out_channels + prev_output_channel, out_channels, time_emb_dim)
        ])
        
        if has_attention:
            self.attentions = nn.ModuleList([
                AttentionBlock(out_channels, context_dim),
                AttentionBlock(out_channels, context_dim),
                AttentionBlock(out_channels, context_dim)
            ])
        
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, res_samples: Tuple[torch.Tensor, ...], 
                time_emb: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            res_x = res_samples[i]
            x = torch.cat([x, res_x], dim=1)
            x = resnet(x, time_emb)
            
            if self.has_attention:
                x = self.attentions[i](x, context)
        
        x = self.upsample(x)
        return x

class OriiO1UNet(nn.Module):
    """U-Net model for diffusion with text conditioning."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_embed_dim = config.block_out_channels[0] * 4
        self.time_embedding = nn.Sequential(
            TimeEmbedding(config.block_out_channels[0]),
            nn.Linear(config.block_out_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input convolution
        self.conv_in = nn.Conv2d(config.in_channels, config.block_out_channels[0], kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        output_channel = config.block_out_channels[0]
        
        for i, down_block_type in enumerate(config.down_block_types):
            input_channel = output_channel
            output_channel = config.block_out_channels[i]
            is_final_block = i == len(config.block_out_channels) - 1
            has_attention = "Attn" in down_block_type
            
            down_block = DownBlock(
                in_channels=input_channel,
                out_channels=output_channel,
                time_emb_dim=time_embed_dim,
                has_attention=has_attention,
                context_dim=config.cross_attention_dim
            )
            self.down_blocks.append(down_block)
        
        # Middle block
        self.mid_block = nn.ModuleList([
            ResnetBlock(config.block_out_channels[-1], config.block_out_channels[-1], time_embed_dim),
            AttentionBlock(config.block_out_channels[-1], config.cross_attention_dim),
            ResnetBlock(config.block_out_channels[-1], config.block_out_channels[-1], time_embed_dim)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(config.block_out_channels))
        
        for i, up_block_type in enumerate(config.up_block_types):
            is_final_block = i == len(config.block_out_channels) - 1
            prev_output_channel = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[min(i + 1, len(config.block_out_channels) - 1)]
            has_attention = "Attn" in up_block_type
            
            up_block = UpBlock(
                in_channels=prev_output_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                time_emb_dim=time_embed_dim,
                has_attention=has_attention,
                context_dim=config.cross_attention_dim
            )
            self.up_blocks.append(up_block)
        
        # Output convolution
        self.conv_norm_out = nn.GroupNorm(32, config.block_out_channels[0])
        self.conv_out = nn.Conv2d(config.block_out_channels[0], config.out_channels, kernel_size=3, padding=1)
        
    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, 
                encoder_hidden_states: Optional[torch.Tensor] = None) -> DiffusionOutput:
        # Time embedding
        time_emb = self.time_embedding(timestep)
        
        # Input convolution
        sample = self.conv_in(sample)
        
        # Downsampling
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(sample, time_emb, encoder_hidden_states)
            down_block_res_samples += res_samples
        
        # Middle block
        for layer in self.mid_block:
            if isinstance(layer, ResnetBlock):
                sample = layer(sample, time_emb)
            else:  # AttentionBlock
                sample = layer(sample, encoder_hidden_states)
        
        # Upsampling
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            sample = up_block(sample, res_samples, time_emb, encoder_hidden_states)
        
        # Output
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)
        
        return DiffusionOutput(sample=sample)

class DDPMScheduler:
    """DDPM noise scheduler for training and inference."""
    
    def __init__(self, config):
        self.num_train_timesteps = config.num_train_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta_schedule = config.beta_schedule
        
        # Create beta schedule
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            self.betas = torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples according to the noise magnitude at each timestep."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Predict the sample at the previous timestep."""
        t = timestep
        
        # Get coefficients
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod_prev[t].to(sample.device) if t > 0 else torch.ones(1).to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Compute predicted original sample
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        
        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t].sqrt() * beta_prod_t_prev / beta_prod_t
        
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        return pred_prev_sample
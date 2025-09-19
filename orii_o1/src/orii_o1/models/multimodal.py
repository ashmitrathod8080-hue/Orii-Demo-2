"""
Orii-O1 Multimodal Model
Combines text and image generation capabilities with cross-modal understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass

from .transformer import OriiO1Transformer
from .diffusion import OriiO1UNet, DDPMScheduler

@dataclass
class MultimodalOutput:
    """Output structure for multimodal model."""
    text_logits: Optional[torch.Tensor] = None
    image_sample: Optional[torch.Tensor] = None
    cross_attention_weights: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

class TextEncoder(nn.Module):
    """Text encoder for conditioning image generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use a smaller transformer for text encoding
        text_config = config.transformer
        text_config.num_layers = 6  # Smaller for efficiency
        
        self.transformer = OriiO1Transformer(text_config)
        self.text_projection = nn.Linear(
            config.transformer.hidden_size, 
            config.diffusion.cross_attention_dim
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text for conditioning."""
        outputs = self.transformer(input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.hidden_states[-1]  # Use last layer
        
        # Project to diffusion cross attention dimension
        text_embeds = self.text_projection(hidden_states)
        
        return text_embeds

class ImageTokenizer(nn.Module):
    """Tokenize images for text generation conditioning."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision encoder (ResNet-like)
        self.patch_embed = nn.Conv2d(3, config.multimodal.image_embed_dim, kernel_size=16, stride=16)
        
        # Position embeddings
        num_patches = (config.diffusion.image_size // 16) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, config.multimodal.image_embed_dim))
        
        # Vision transformer layers
        self.vision_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.multimodal.image_embed_dim,
                nhead=8,
                dim_feedforward=config.multimodal.image_embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])
        
        # Project to text embedding space
        self.image_projection = nn.Linear(
            config.multimodal.image_embed_dim,
            config.transformer.hidden_size
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Tokenize images for text conditioning."""
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # B, D, H/16, W/16
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Vision transformer
        for layer in self.vision_layers:
            x = layer(x)
        
        # Project to text space
        image_embeds = self.image_projection(x)
        
        return image_embeds

class CrossModalFusion(nn.Module):
    """Cross-modal fusion layer for text-image understanding."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.text_dim = config.transformer.hidden_size
        self.image_dim = config.multimodal.image_embed_dim
        self.fusion_dim = config.multimodal.fusion_dim
        
        # Text and image projections
        self.text_proj = nn.Linear(self.text_dim, self.fusion_dim)
        self.image_proj = nn.Linear(self.image_dim, self.fusion_dim)
        
        # Cross attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.multimodal.cross_attention_layers)
        ])
        
        # Output projections
        self.text_out_proj = nn.Linear(self.fusion_dim, self.text_dim)
        self.image_out_proj = nn.Linear(self.fusion_dim, self.image_dim)
        
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform cross-modal fusion."""
        # Project to fusion space
        text_proj = self.text_proj(text_embeds)
        image_proj = self.image_proj(image_embeds)
        
        # Cross attention
        text_fused = text_proj
        image_fused = image_proj
        
        for layer in self.cross_attention_layers:
            # Text attends to image
            text_attended, _ = layer(text_fused, image_fused, image_fused)
            text_fused = text_fused + text_attended
            
            # Image attends to text
            image_attended, _ = layer(image_fused, text_fused, text_fused)
            image_fused = image_fused + image_attended
        
        # Project back to original spaces
        text_out = self.text_out_proj(text_fused)
        image_out = self.image_out_proj(image_fused)
        
        return text_out, image_out

class OriiO1Multimodal(nn.Module):
    """Main multimodal model combining text and image generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core models
        self.text_model = OriiO1Transformer(config.transformer)
        self.image_model = OriiO1UNet(config.diffusion)
        
        # Conditioning models
        self.text_encoder = TextEncoder(config)
        self.image_tokenizer = ImageTokenizer(config)
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(config)
        
        # Noise scheduler for diffusion
        self.scheduler = DDPMScheduler(config.diffusion)
        
        # Special tokens for multimodal interaction
        self.image_start_token = config.transformer.vocab_size - 3
        self.image_end_token = config.transformer.vocab_size - 2
        self.multimodal_token = config.transformer.vocab_size - 1
        
    def generate_text(
        self,
        input_ids: torch.Tensor,
        image_condition: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **generation_kwargs
    ) -> torch.Tensor:
        """Generate text with optional image conditioning."""
        if image_condition is not None:
            # Tokenize image for conditioning
            image_embeds = self.image_tokenizer(image_condition)
            
            # Get text embeddings
            text_embeds = self.text_model.token_embeddings(input_ids)
            
            # Cross-modal fusion
            fused_text_embeds, _ = self.cross_modal_fusion(text_embeds, image_embeds)
            
            # Replace text embeddings in the model temporarily
            original_embeddings = self.text_model.token_embeddings.weight.data.clone()
            self.text_model.token_embeddings.weight.data[:fused_text_embeds.shape[1]] = fused_text_embeds[0]
        
        try:
            # Generate text
            generated = self.text_model.generate(input_ids, max_new_tokens=max_new_tokens, **generation_kwargs)
            return generated
        finally:
            # Restore original embeddings
            if image_condition is not None:
                self.text_model.token_embeddings.weight.data = original_embeddings
    
    def generate_image(
        self,
        prompt_embeds: torch.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generate image from text embeddings."""
        device = prompt_embeds.device
        batch_size = prompt_embeds.shape[0]
        
        # Initialize random noise
        shape = (batch_size, self.config.diffusion.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(
            self.config.diffusion.num_train_timesteps - 1, 0, num_inference_steps, 
            dtype=torch.long, device=device
        )
        
        # Denoising loop
        for t in timesteps:
            # Expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Predict noise
            noise_pred = self.image_model(
                latent_model_input,
                t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=prompt_embeds
            ).sample
            
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t.item(), latents)
        
        return latents
    
    def encode_text_for_image(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text for image generation conditioning."""
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(
        self,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        mode: str = "text"  # "text", "image", or "multimodal"
    ) -> MultimodalOutput:
        """Forward pass for training."""
        
        if mode == "text":
            # Text generation forward pass
            if images is not None:
                # Text generation with image conditioning
                image_embeds = self.image_tokenizer(images)
                text_embeds = self.text_model.token_embeddings(text_input_ids)
                
                # Cross-modal fusion
                fused_text_embeds, _ = self.cross_modal_fusion(text_embeds, image_embeds)
                
                # Forward through text model with fused embeddings
                outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
                return MultimodalOutput(text_logits=outputs.logits)
            else:
                # Standard text generation
                outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
                return MultimodalOutput(text_logits=outputs.logits)
        
        elif mode == "image":
            # Image generation forward pass
            if text_input_ids is not None:
                # Get text conditioning
                text_embeds = self.encode_text_for_image(text_input_ids, text_attention_mask)
            else:
                text_embeds = None
            
            # Add noise to images for training
            if noise is None:
                noise = torch.randn_like(images)
            
            noisy_images = self.scheduler.add_noise(images, noise, image_timesteps)
            
            # Predict noise
            noise_pred = self.image_model(
                noisy_images,
                image_timesteps,
                encoder_hidden_states=text_embeds
            ).sample
            
            return MultimodalOutput(image_sample=noise_pred)
        
        elif mode == "multimodal":
            # Joint text-image understanding
            text_embeds = self.text_model.token_embeddings(text_input_ids)
            image_embeds = self.image_tokenizer(images)
            
            # Cross-modal fusion
            fused_text_embeds, fused_image_embeds = self.cross_modal_fusion(text_embeds, image_embeds)
            
            # Text generation with fused embeddings
            text_outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
            
            return MultimodalOutput(text_logits=text_outputs.logits)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

class SafetyFilter(nn.Module):
    """Content safety filter for generated content."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.threshold = config.safety_threshold
        
        # Simple text classifier
        self.text_classifier = nn.Sequential(
            nn.Linear(config.transformer.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Simple image classifier
        self.image_classifier = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def filter_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Filter text content for safety."""
        # Average pooling over sequence length
        pooled_embeds = text_embeds.mean(dim=1)
        safety_score = self.text_classifier(pooled_embeds)
        return safety_score
    
    def filter_image(self, images: torch.Tensor) -> torch.Tensor:
        """Filter image content for safety."""
        safety_score = self.image_classifier(images)
        return safety_score
    
    def is_safe_text(self, text_embeds: torch.Tensor) -> bool:
        """Check if text content is safe."""
        safety_score = self.filter_text(text_embeds)
        return (safety_score < self.threshold).all().item()
    
    def is_safe_image(self, images: torch.Tensor) -> bool:
        """Check if image content is safe."""
        safety_score = self.filter_image(images)
        return (safety_score < self.threshold).all().item()

class HumanLikeResponseProcessor:
    """Post-processor to make responses more human-like."""
    
    def __init__(self, config):
        self.config = config
        
        # Human-like response patterns
        self.conversation_starters = [
            "I think", "In my opinion", "From my perspective", "I believe",
            "It seems to me", "I feel like", "I'd say", "Personally"
        ]
        
        self.hesitation_markers = [
            "um", "well", "you know", "actually", "I mean", "sort of", "kind of"
        ]
        
        self.emphasis_words = [
            "really", "definitely", "absolutely", "totally", "completely", "quite"
        ]
    
    def add_human_touches(self, text: str) -> str:
        """Add human-like elements to generated text."""
        import random
        
        # Add occasional hesitation markers
        if random.random() < 0.2:
            starter = random.choice(self.conversation_starters)
            text = f"{starter}, {text.lower()}"
        
        # Add emphasis occasionally
        if random.random() < 0.15:
            words = text.split()
            if len(words) > 3:
                pos = random.randint(1, len(words) - 2)
                emphasis = random.choice(self.emphasis_words)
                words.insert(pos, emphasis)
                text = " ".join(words)
        
        return text
    
    def vary_response_length(self, text: str, target_variety: float = 0.3) -> str:
        """Vary response length for more natural conversation."""
        import random
        
        sentences = text.split('. ')
        if len(sentences) > 1 and random.random() < target_variety:
            # Randomly keep 70-90% of sentences
            keep_ratio = random.uniform(0.7, 0.9)
            keep_count = max(1, int(len(sentences) * keep_ratio))
            sentences = sentences[:keep_count]
            text = '. '.join(sentences)
            
            if not text.endswith('.'):
                text += '.'
        
        return text

# Utility functions for model integration
def load_pretrained_components(config, device="auto"):
    """Load pretrained components if available."""
    components = {}
    
    try:
        # Try to load pretrained tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./models/pretrained")
        tokenizer.pad_token = tokenizer.eos_token
        components['tokenizer'] = tokenizer
    except:
        components['tokenizer'] = None
    
    return components

def setup_model_for_inference(model: OriiO1Multimodal, device: str, dtype: str = "float16"):
    """Setup model for efficient inference."""
    # Move to device
    model = model.to(device)
    
    # Set precision
    if dtype == "float16":
        model = model.half()
    elif dtype == "bfloat16":
        model = model.bfloat16()
    
    # Enable eval mode
    model.eval()
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass
    
    return model

def create_attention_mask_for_multimodal(text_length: int, image_length: int, device: torch.device) -> torch.Tensor:
    """Create attention mask for multimodal inputs."""
    total_length = text_length + image_length
    
    # Create causal mask for text, bidirectional for image tokens
    mask = torch.zeros(total_length, total_length, device=device)
    
    # Text can only attend to previous text tokens
    for i in range(text_length):
        mask[i, :i+1] = 1
    
    # Image tokens can attend to all text and image tokens
    mask[text_length:, :] = 1
    
    # Convert to attention mask format
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    attention_mask = (1.0 - mask) * -10000.0
    
    return attention_mask
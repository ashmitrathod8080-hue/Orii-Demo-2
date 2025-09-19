"""
Orii-O1 Model Configuration
Defines the architecture and parameters for both text and image generation models.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TransformerConfig:
    """Configuration for the text generation transformer model."""
    vocab_size: int = 50000
    max_seq_length: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 10000.0
    use_cache: bool = True
    
    # Human-like response configuration
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0

@dataclass
class DiffusionConfig:
    """Configuration for the image generation diffusion model."""
    image_size: int = 512
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: tuple = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    up_block_types: tuple = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    attention_head_dim: int = 64
    cross_attention_dim: int = 768
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = "blurry, bad quality, distorted, deformed"

@dataclass
class MultimodalConfig:
    """Configuration for multimodal integration."""
    text_embed_dim: int = 768
    image_embed_dim: int = 768
    cross_attention_layers: int = 4
    fusion_dim: int = 1024
    
@dataclass
class OriiO1Config:
    """Main configuration class for Orii-O1 model."""
    model_name: str = "orii-o1"
    version: str = "1.0.0"
    
    # Model configurations
    transformer: TransformerConfig = None
    diffusion: DiffusionConfig = None
    multimodal: MultimodalConfig = None
    
    # Device and precision
    device: str = "auto"  # auto, cpu, cuda, mps
    dtype: str = "float16"  # float32, float16, bfloat16
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    offload_to_cpu: bool = False
    
    # Safety and filtering
    content_filter: bool = True
    safety_threshold: float = 0.7
    
    def __post_init__(self):
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if self.diffusion is None:
            self.diffusion = DiffusionConfig()
        if self.multimodal is None:
            self.multimodal = MultimodalConfig()
            
        # Auto device selection
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "transformer": self.transformer.__dict__,
            "diffusion": self.diffusion.__dict__,
            "multimodal": self.multimodal.__dict__,
            "device": self.device,
            "dtype": self.dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_flash_attention": self.use_flash_attention,
            "offload_to_cpu": self.offload_to_cpu,
            "content_filter": self.content_filter,
            "safety_threshold": self.safety_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        transformer_config = TransformerConfig(**config_dict.get("transformer", {}))
        diffusion_config = DiffusionConfig(**config_dict.get("diffusion", {}))
        multimodal_config = MultimodalConfig(**config_dict.get("multimodal", {}))
        
        return cls(
            model_name=config_dict.get("model_name", "orii-o1"),
            version=config_dict.get("version", "1.0.0"),
            transformer=transformer_config,
            diffusion=diffusion_config,
            multimodal=multimodal_config,
            device=config_dict.get("device", "auto"),
            dtype=config_dict.get("dtype", "float16"),
            gradient_checkpointing=config_dict.get("gradient_checkpointing", True),
            use_flash_attention=config_dict.get("use_flash_attention", True),
            offload_to_cpu=config_dict.get("offload_to_cpu", False),
            content_filter=config_dict.get("content_filter", True),
            safety_threshold=config_dict.get("safety_threshold", 0.7)
        )

# Predefined model sizes
ORII_O1_SMALL = OriiO1Config(
    transformer=TransformerConfig(
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=2048
    )
)

ORII_O1_MEDIUM = OriiO1Config(
    transformer=TransformerConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072
    )
)

ORII_O1_LARGE = OriiO1Config(
    transformer=TransformerConfig(
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        intermediate_size=4096
    )
)
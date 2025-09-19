#!/usr/bin/env python3
"""
Orii-O1 Training Script
Main script for training the multimodal model.
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orii_o1.models.multimodal import OriiO1Multimodal
from orii_o1.training.trainer import OriiO1Trainer, TrainingConfig
from orii_o1.training.data_loader import create_data_loaders
from config.model_config import OriiO1Config, ORII_O1_SMALL, ORII_O1_MEDIUM, ORII_O1_LARGE

def parse_args():
    parser = argparse.ArgumentParser(description="Train Orii-O1 Model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_size", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--mode", type=str, default="multimodal", choices=["text", "image", "multimodal"])
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # System arguments
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./models/checkpoints")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    
    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="orii-o1")
    
    # Model configuration
    parser.add_argument("--config_file", type=str, help="Path to model config YAML file")
    
    return parser.parse_args()

def load_config(config_file: str = None, model_size: str = "medium") -> OriiO1Config:
    """Load model configuration."""
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
        return OriiO1Config.from_dict(config_dict)
    else:
        # Use predefined configurations
        if model_size == "small":
            return ORII_O1_SMALL
        elif model_size == "medium":
            return ORII_O1_MEDIUM
        elif model_size == "large":
            return ORII_O1_LARGE
        else:
            raise ValueError(f"Unknown model size: {model_size}")

def setup_tokenizer():
    """Setup tokenizer for training."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./models/pretrained")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        print("Warning: transformers not available, using simple tokenizer")
        return None

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config_file, args.model_size)
    
    print(f"Orii-O1 Training")
    print(f"Model size: {args.model_size}")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Device: {config.device}")
    print(f"Mixed precision: {args.mixed_precision}")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    if tokenizer:
        print(f"Using tokenizer: {tokenizer.__class__.__name__}")
        # Update vocab size in config
        config.transformer.vocab_size = tokenizer.vocab_size
    
    # Create model
    print("Creating model...")
    model = OriiO1Multimodal(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        mode=args.mode,
        num_workers=args.num_workers,
        tokenizer=tokenizer
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        use_wandb=args.use_wandb,
        project_name=args.project_name
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = OriiO1Trainer(model, training_config)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
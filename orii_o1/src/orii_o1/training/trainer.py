"""
Orii-O1 Training Module
Handles training for both text and image generation components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import os
import json
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time

from ..models.multimodal import OriiO1Multimodal, SafetyFilter
from .data_loader import OriiO1Dataset, create_data_loaders
from .optimizer import create_optimizer_and_scheduler

@dataclass
class TrainingConfig:
    """Training configuration."""
    # General training
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Mixed precision
    mixed_precision: str = "fp16"  # fp16, bf16, no
    gradient_accumulation_steps: int = 4
    
    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "./models/checkpoints"
    
    # Logging
    log_every: int = 100
    use_wandb: bool = True
    project_name: str = "orii-o1"
    
    # Safety
    enable_safety_filter: bool = True
    safety_filter_frequency: int = 100
    
    # Model specific
    text_loss_weight: float = 1.0
    image_loss_weight: float = 1.0
    multimodal_loss_weight: float = 0.5

class OriiO1Trainer:
    """Main trainer class for Orii-O1 model."""
    
    def __init__(self, model: OriiO1Multimodal, config: TrainingConfig, accelerator: Optional[Accelerator] = None):
        self.model = model
        self.config = config
        
        # Initialize accelerator
        self.accelerator = accelerator or Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None,
            project_config={"project_name": config.project_name} if config.use_wandb else None
        )
        
        # Setup model
        self.model = self.accelerator.prepare(self.model)
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model, 
            config.learning_rate,
            config.weight_decay,
            config.warmup_steps
        )
        
        # Prepare optimizer and scheduler
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        
        # Safety filter
        if config.enable_safety_filter:
            self.safety_filter = SafetyFilter(model.config)
            self.safety_filter = self.accelerator.prepare(self.safety_filter)
        else:
            self.safety_filter = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Logging
        if self.accelerator.is_main_process:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(config.checkpoint_dir, "logs"))
        
        # Initialize wandb
        if config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.project_name,
                config=vars(config)
            )
    
    def compute_text_loss(self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for text generation."""
        # Shift labels for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_attention_mask = shift_attention_mask.view(-1)
        
        # Compute loss only on non-padded tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        
        # Apply attention mask
        loss = loss * shift_attention_mask
        
        # Average over valid tokens
        return loss.sum() / shift_attention_mask.sum().clamp(min=1)
    
    def compute_image_loss(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss for image diffusion."""
        return F.mse_loss(noise_pred, noise_target)
    
    def compute_safety_loss(self, text_embeds: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Compute safety loss to encourage safe content generation."""
        if self.safety_filter is None:
            return torch.tensor(0.0, device=text_embeds.device)
        
        text_safety = self.safety_filter.filter_text(text_embeds)
        image_safety = self.safety_filter.filter_image(images)
        
        # Encourage low safety scores (safe content)
        safety_loss = text_safety.mean() + image_safety.mean()
        return safety_loss
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single training step."""
        self.model.train()
        
        # Prepare batch
        text_input_ids = batch.get('text_input_ids')
        text_attention_mask = batch.get('text_attention_mask')
        text_labels = batch.get('text_labels')
        images = batch.get('images')
        image_timesteps = batch.get('image_timesteps')
        noise = batch.get('noise')
        
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        losses = {}
        
        # Text generation loss
        if text_input_ids is not None and text_labels is not None:
            text_outputs = self.model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                mode="text"
            )
            
            text_loss = self.compute_text_loss(
                text_outputs.text_logits,
                text_labels,
                text_attention_mask
            )
            
            total_loss += self.config.text_loss_weight * text_loss
            losses['text_loss'] = text_loss
        
        # Image generation loss
        if images is not None and image_timesteps is not None and noise is not None:
            image_outputs = self.model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                images=images,
                image_timesteps=image_timesteps,
                noise=noise,
                mode="image"
            )
            
            image_loss = self.compute_image_loss(
                image_outputs.image_sample,
                noise
            )
            
            total_loss += self.config.image_loss_weight * image_loss
            losses['image_loss'] = image_loss
        
        # Multimodal understanding loss
        if text_input_ids is not None and images is not None:
            multimodal_outputs = self.model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                images=images,
                mode="multimodal"
            )
            
            if text_labels is not None:
                multimodal_loss = self.compute_text_loss(
                    multimodal_outputs.text_logits,
                    text_labels,
                    text_attention_mask
                )
                
                total_loss += self.config.multimodal_loss_weight * multimodal_loss
                losses['multimodal_loss'] = multimodal_loss
        
        # Safety loss (optional)
        if self.safety_filter is not None and self.global_step % self.config.safety_filter_frequency == 0:
            if text_input_ids is not None and images is not None:
                text_embeds = self.model.text_model.token_embeddings(text_input_ids)
                safety_loss = self.compute_safety_loss(text_embeds, images)
                total_loss += 0.1 * safety_loss  # Small weight for safety
                losses['safety_loss'] = safety_loss
        
        losses['total_loss'] = total_loss
        
        # Backward pass
        self.accelerator.backward(total_loss)
        
        return losses
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Similar to training step but without backward pass
            text_input_ids = batch.get('text_input_ids')
            text_attention_mask = batch.get('text_attention_mask')
            text_labels = batch.get('text_labels')
            images = batch.get('images')
            image_timesteps = batch.get('image_timesteps')
            noise = batch.get('noise')
            
            total_loss = torch.tensor(0.0, device=self.accelerator.device)
            losses = {}
            
            # Text generation validation
            if text_input_ids is not None and text_labels is not None:
                text_outputs = self.model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    mode="text"
                )
                
                text_loss = self.compute_text_loss(
                    text_outputs.text_logits,
                    text_labels,
                    text_attention_mask
                )
                
                total_loss += text_loss
                losses['val_text_loss'] = text_loss
            
            # Image generation validation
            if images is not None and image_timesteps is not None and noise is not None:
                image_outputs = self.model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    images=images,
                    image_timesteps=image_timesteps,
                    noise=noise,
                    mode="image"
                )
                
                image_loss = self.compute_image_loss(
                    image_outputs.image_sample,
                    noise
                )
                
                total_loss += image_loss
                losses['val_image_loss'] = image_loss
            
            losses['val_total_loss'] = total_loss
        
        return losses
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Training Epoch {self.current_epoch}",
            disable=not self.accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            losses = self.training_step(batch)
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())
            
            # Logging
            if self.global_step % self.config.log_every == 0 and self.accelerator.is_main_process:
                lr = self.scheduler.get_last_lr()[0]
                
                # Log to tensorboard
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', lr, self.global_step)
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb_dict = {f'train/{key}': value.item() for key, value in losses.items()}
                    wandb_dict['train/learning_rate'] = lr
                    wandb_dict['train/global_step'] = self.global_step
                    self.accelerator.log(wandb_dict)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average losses
        avg_losses = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses = {}
        
        progress_bar = tqdm(
            val_loader,
            desc="Validation",
            disable=not self.accelerator.is_main_process
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                losses = self.validation_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item())
        
        # Average losses
        avg_losses = {key: sum(values) / len(values) for key, values in val_losses.items()}
        
        # Log validation results
        if self.accelerator.is_main_process:
            for key, value in avg_losses.items():
                self.writer.add_scalar(f'val/{key}', value, self.global_step)
            
            if self.config.use_wandb:
                wandb_dict = {f'val/{key}': value for key, value in avg_losses.items()}
                self.accelerator.log(wandb_dict)
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # Load model state
        self.accelerator.load_state(checkpoint_path)
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        print("Starting training...")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Total steps: {len(train_loader) * self.config.num_epochs}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = {}
            if val_loader is not None and epoch % (self.config.eval_every // len(train_loader)) == 0:
                val_losses = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch} Summary:")
                print(f"Time: {epoch_time:.2f}s")
                for key, value in train_losses.items():
                    print(f"Train {key}: {value:.4f}")
                for key, value in val_losses.items():
                    print(f"Val {key}: {value:.4f}")
            
            # Check if this is the best model
            current_loss = val_losses.get('val_total_loss', train_losses.get('total_loss', float('inf')))
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            # Save checkpoint
            if (epoch + 1) % (self.config.save_every // len(train_loader)) == 0:
                self.save_checkpoint(epoch, is_best)
        
        # Final checkpoint
        if self.accelerator.is_main_process:
            self.save_checkpoint(self.config.num_epochs - 1, is_best=False)
            print("Training completed!")
        
        # Close logging
        if self.accelerator.is_main_process:
            self.writer.close()
            if self.config.use_wandb:
                wandb.finish()

def create_trainer(model: OriiO1Multimodal, config: TrainingConfig) -> OriiO1Trainer:
    """Create and setup trainer."""
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.use_wandb else None
    )
    
    trainer = OriiO1Trainer(model, config, accelerator)
    return trainer
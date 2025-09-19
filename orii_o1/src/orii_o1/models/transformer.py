"""
Orii-O1 Transformer Model
A custom transformer architecture optimized for human-like text generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ModelOutput:
    """Output structure for model forward pass."""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache cos/sin values
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int):
        """Apply rotary position embedding."""
        cos = self.cos_cached[:seq_len, :].to(x.device)
        sin = self.sin_cached[:seq_len, :].to(x.device)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and optional flash attention."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout_prob
        
        assert self.hidden_size % self.num_heads == 0
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, 
            max_seq_len=config.max_seq_length,
            theta=config.rope_theta
        )
        
        self.dropout = nn.Dropout(self.attention_dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handle past key values for caching
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        # Store for next iteration if using cache
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
        
        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value

class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU requires 2/3 scaling for intermediate size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Pre-norm attention
        normed_hidden_states = self.ln1(hidden_states)
        attn_output, present_key_value = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-norm feed forward
        normed_hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states, present_key_value

class OriiO1Transformer(nn.Module):
    """Main Orii-O1 Transformer model for text generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings and lm_head weights
        self.lm_head.weight = self.token_embeddings.weight
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create causal attention mask."""
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Create padding mask
        padding_mask = (input_ids == self.config.pad_token_id).unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Combine masks
        attention_mask = causal_mask | padding_mask
        
        # Convert to additive mask
        attention_mask = attention_mask.to(dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        
        return attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, ModelOutput]:
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Initialize past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Forward through transformer layers
        all_hidden_states = () if return_dict else None
        all_attentions = () if return_dict else None
        next_decoder_cache = () if use_cache else None
        
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if return_dict:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                next_decoder_cache = next_decoder_cache + (present_key_value,)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if return_dict:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return (logits, next_decoder_cache, all_hidden_states, all_attentions)
        
        return ModelOutput(
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        batch_size = input_ids.shape[0]
        generated_tokens = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]  # Get last token logits
                past_key_values = outputs.past_key_values
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token in set(generated_tokens[i].tolist()):
                            if logits[i, token] < 0:
                                logits[i, token] *= repetition_penalty
                            else:
                                logits[i, token] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_logits[..., -1:]] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if (next_token == eos_token_id).all():
                    break
                
                # Update sequences
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                input_ids = next_token
        
        return generated_tokens
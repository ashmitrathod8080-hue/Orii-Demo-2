"""
Orii-O1 Web Interface
Flask web application for interacting with the model.
"""

import os
import sys
import torch
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import json
from datetime import datetime

# Simple fallback config class
class OriiO1Config:
    def __init__(self):
        self.dtype = "float32"
        self.hidden_size = 768
        self.num_layers = 12
        self.vocab_size = 50257
        self.max_position_embeddings = 1024
    
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

# Simple multimodal model implementation
class OriiO1Multimodal:
    def __init__(self, config):
        self.config = config
        # Try to use actual transformers if available
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
            gpt_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_positions=config.max_position_embeddings,
                n_embd=config.hidden_size,
                n_layer=config.num_layers,
                n_head=12
            )
            self.text_model = GPT2LMHeadModel(gpt_config)
        except ImportError:
            # Fallback to simple linear layer
            import torch.nn as nn
            self.text_model = nn.Sequential(
                nn.Embedding(config.vocab_size, config.hidden_size),
                nn.Linear(config.hidden_size, config.vocab_size)
            )
        
        self.device = "cpu"
    
    def to(self, device):
        self.device = device
        self.text_model = self.text_model.to(device)
        return self
    
    def eval(self):
        self.text_model.eval()
        return self
    
    def half(self):
        self.text_model = self.text_model.half()
        return self
    
    def parameters(self):
        return self.text_model.parameters()
    
    def load_state_dict(self, state_dict, strict=True):
        try:
            self.text_model.load_state_dict(state_dict, strict=False)
        except:
            print("Warning: Could not load state dict completely")
    
    def generate_text(self, input_ids, image_condition=None, max_new_tokens=150, 
                     temperature=0.8, top_p=0.9, do_sample=True, **kwargs):
        try:
            if hasattr(self.text_model, 'generate'):
                return self.text_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=50256,
                    **kwargs
                )
            else:
                # Simple fallback generation
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                new_tokens = torch.randint(0, self.config.vocab_size, 
                                         (batch_size, max_new_tokens), 
                                         device=input_ids.device)
                return torch.cat([input_ids, new_tokens], dim=1)
        except Exception as e:
            print(f"Generation error: {e}")
            return input_ids
    
    def encode_text_for_image(self, input_ids, attention_mask=None):
        # Mock text encoding for image generation
        return torch.randn(1, self.config.hidden_size, device=self.device)
    
    def generate_image(self, text_embeds, height=512, width=512, 
                      num_inference_steps=30, guidance_scale=7.5):
        # Mock image generation - return random image tensor
        return torch.randn(1, 3, height, width, device=self.device)

class HumanLikeResponseProcessor:
    def __init__(self, config):
        self.config = config
    
    def add_human_touches(self, response):
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        return response
    
    def vary_response_length(self, response):
        return response

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Global model variables
model = None
tokenizer = None
config = None
response_processor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_tokenizer():
    """Setup tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return None

def load_model(model_path: str, config_path: str = None):
    """Load the trained model."""
    global model, config, response_processor
    
    print(f"Attempting to load model from {model_path}...")
    
    # Create default config
    config = OriiO1Config()
    
    try:
        # Try to load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                    config = OriiO1Config.from_dict(config_dict)
                    print("Loaded config from file")
                except ImportError:
                    try:
                        config_dict = json.load(f)
                        config = OriiO1Config.from_dict(config_dict)
                        print("Loaded config from JSON file")
                    except:
                        print("Could not parse config file, using defaults")
        
        # Try to load model checkpoint if it exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            print("Loaded checkpoint file")
            
            # Update config from checkpoint if available
            if "config" in checkpoint:
                checkpoint_config = checkpoint["config"]
                if hasattr(checkpoint_config, '__dict__'):
                    for key, value in checkpoint_config.__dict__.items():
                        setattr(config, key, value)
                elif isinstance(checkpoint_config, dict):
                    config = OriiO1Config.from_dict(checkpoint_config)
        else:
            print(f"Model file {model_path} not found, creating new model")
            checkpoint = None
        
        # Create model
        model = OriiO1Multimodal(config)
        
        # Load state dict if available
        if checkpoint:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded model state dict")
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print("Loaded state dict")
            else:
                # Assume the entire checkpoint is the state dict
                try:
                    model.load_state_dict(checkpoint)
                    print("Loaded checkpoint as state dict")
                except:
                    print("Could not load checkpoint as state dict")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model instance with default config")
        config = OriiO1Config()
        model = OriiO1Multimodal(config)
    
    # Setup for inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = model.to(device)
        model.eval()
        
        if hasattr(config, 'dtype') and config.dtype == "float16" and device == "cuda":
            model = model.half()
            print("Using half precision")
    except Exception as e:
        print(f"Error setting up model device: {e}")
    
    # Setup response processor
    response_processor = HumanLikeResponseProcessor(config)
    
    print("Model setup complete")

def process_image(image_file):
    """Process uploaded image."""
    try:
        from torchvision import transforms
        
        # Open and convert image
        image = Image.open(image_file.stream).convert("RGB")
        
        # Transform image
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        if torch.cuda.is_available() and model.device == "cuda":
            image_tensor = image_tensor.cuda()
        
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def generate_text_response(prompt: str, image_tensor=None, max_tokens=150):
    """Generate text response from the model."""
    global model, tokenizer, response_processor
    
    if model is None:
        return "Model not loaded. Please check the model path and try again."
    
    try:
        # Tokenize input
        if tokenizer:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, 
                             truncation=True, padding=True)
            if model.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
        else:
            # Simple character-based encoding as fallback
            tokens = [min(ord(c), config.vocab_size-1) for c in prompt[:512]]
            input_ids = torch.tensor([tokens], dtype=torch.long)
            if model.device == "cuda":
                input_ids = input_ids.cuda()
        
        # Generate response
        with torch.no_grad():
            try:
                if image_tensor is not None:
                    # Multimodal generation
                    generated = model.generate_text(
                        input_ids,
                        image_condition=image_tensor,
                        max_new_tokens=max_tokens,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True
                    )
                else:
                    # Text-only generation
                    generated = model.generate_text(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True
                    )
            except Exception as gen_error:
                print(f"Generation error: {gen_error}")
                # Fallback response
                return "I understand your message. How can I help you today?"
        
        # Decode response
        if tokenizer:
            try:
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                # Remove input prompt from response
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                if response.startswith(input_text):
                    response = response[len(input_text):].strip()
            except Exception as decode_error:
                print(f"Decode error: {decode_error}")
                response = "I'm processing your request..."
        else:
            # Simple character-based decoding
            try:
                response_tokens = generated[0][input_ids.shape[1]:].cpu().numpy()
                response = "".join([chr(min(int(token), 127)) for token in response_tokens if token > 0])
            except:
                response = "I'm here to help you."
        
        # Clean up response
        response = response.strip()
        if not response or len(response) < 5:
            response = "I understand. Could you please provide more details about what you'd like to know?"
        
        # Apply human-like processing
        if response_processor:
            response = response_processor.add_human_touches(response)
            response = response_processor.vary_response_length(response)
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response. Could you please try again?"

def generate_image_response(prompt: str, width=512, height=512):
    """Generate image from text prompt."""
    global model, tokenizer
    
    if model is None:
        return None
    
    try:
        # Encode text
        if tokenizer:
            text_inputs = tokenizer(
                prompt, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            if model.device == "cuda":
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
            
            text_embeds = model.encode_text_for_image(
                text_inputs["input_ids"], 
                text_inputs.get("attention_mask")
            )
        else:
            tokens = [min(ord(c), config.vocab_size-1) for c in prompt[:77]]
            input_ids = torch.tensor([tokens], dtype=torch.long)
            if model.device == "cuda":
                input_ids = input_ids.cuda()
            text_embeds = model.encode_text_for_image(input_ids)
        
        # Generate image
        with torch.no_grad():
            image_latents = model.generate_image(
                text_embeds,
                height=height,
                width=width,
                num_inference_steps=30,
                guidance_scale=7.5
            )
        
        # Convert to PIL Image
        image_latents = (image_latents + 1) * 0.5  # Convert from [-1,1] to [0,1]
        image_latents = torch.clamp(image_latents, 0, 1)
        
        # Convert to numpy and then PIL
        image_np = image_latents.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype('uint8')
        image_pil = Image.fromarray(image_np)
        
        # Convert to base64 for web display
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Chat interface page."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat API endpoint."""
    try:
        data = request.json
        message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Build conversation context
        context_parts = []
        for entry in conversation_history[-6:]:  # Last 3 exchanges
            role = entry.get('role')
            content = entry.get('content')
            if role and content:
                context_parts.append(f"{role}: {content}")
        
        context_parts.append(f"user: {message}")
        context_parts.append("assistant:")
        context = "\n".join(context_parts)
        
        # Generate response
        response = generate_text_response(context, max_tokens=200)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/multimodal', methods=['POST'])
def api_multimodal():
    """Multimodal chat API endpoint."""
    try:
        message = request.form.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process image if provided
        image_tensor = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                image_tensor = process_image(file)
        
        # Generate response
        response = generate_text_response(message, image_tensor, max_tokens=200)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_image', methods=['POST'])
def api_generate_image():
    """Image generation API endpoint."""
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Generate image
        image_data = generate_image_response(prompt, width, height)
        
        if image_data:
            return jsonify({
                'image': image_data,
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to generate image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """Get model status."""
    global model
    
    status = {
        'model_loaded': model is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available()
    }
    
    if model is not None:
        try:
            status['model_device'] = str(model.device)
            params = list(model.parameters())
            if params:
                status['model_dtype'] = str(params[0].dtype)
            else:
                status['model_dtype'] = 'unknown'
        except Exception as e:
            status['model_device'] = 'unknown'
            status['model_dtype'] = 'unknown'
            status['status_error'] = str(e)
    
    return jsonify(status)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

def create_app(model_path: str, config_path: str = None, debug: bool = False):
    """Create and configure the Flask app."""
    global tokenizer
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load model
    load_model(model_path, config_path)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    
    # Configure app
    app.debug = debug
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Orii-O1 Web Interface")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(args.model_path, args.config_path, args.debug)
    
    # Run app
    print(f"Starting Orii-O1 Web Interface on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
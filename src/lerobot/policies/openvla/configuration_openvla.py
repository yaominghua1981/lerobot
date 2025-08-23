from dataclasses import dataclass

from lerobot.configs.policies import PolicyConfig
import os

@dataclass
class OpenVLAConfig(PolicyConfig):
    # Model Architecture
    pretrained_model_name_or_path: str = "openvla/openvla-7b"
    vlm_model_name: str = "openvla/openvla-7b"
    
    # Training Configuration
    chunk_size: int = 10
    n_obs_steps: int = 1
    n_action_steps: int = 10
    num_steps: int = 50
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = True
    train_action_head_only: bool = False
    
    # Model Dimensions
    max_action_dim: int = 14  # Max action dimension across all supported robots
    max_state_dim: int = 14   # Max state dimension
    hidden_size: int = 4096   # OpenVLA hidden size
    
    # Image Processing
    image_resolution: tuple = (224, 224)
    resize_imgs_with_padding: tuple = (224, 224)
    
    # Tokenizer Configuration
    tokenizer_max_length: int = 512
    pad_language_to: str = "max_length"
    add_image_special_tokens: bool = True
    
    # Normalization Parameters
    action_norm_min: float = -1.0
    action_norm_max: float = 1.0
    img_norm_mean: float = 0.5
    img_norm_std: float = 0.5
    
    # Attention Configuration
    attention_mode: str = "full"
    use_cache: bool = True
    
    # Sampling Parameters
    min_period: float = 0.01
    max_period: float = 10.0
    
    # Device Configuration
    quantize_for_jetson: bool = True  # Enable quantization for Jetson Orin
    
    # Dataset Adaptation
    adapt_to_pi_aloha: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        # Validate essential configurations
        if not os.path.exists(self.pretrained_model_name_or_path):
            # Check if it's a HuggingFace model name
            if not self.pretrained_model_name_or_path.startswith("openvla/"):
                raise ValueError(f"Model path {self.pretrained_model_name_or_path} not found!")
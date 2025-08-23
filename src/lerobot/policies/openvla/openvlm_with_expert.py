"""
OpenVLA with Expert Integration Module

This integrates the original OpenVLA approach with expert demonstrations and guidance,
providing compatibility with expert-driven training and fine-tuning strategies.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy, OpenVLAModel


class OpenVLAWithExpert(nn.Module):
    """
    OpenVLA model enhanced with expert demonstrations for better training.
    
    This class provides mechanisms to:
    1. Leverage expert demonstrations for action prediction
    2. Apply behavioral cloning with expert data
    3. Implement imitation learning from expert trajectories
    4. Provide expert guidance during training
    """
    
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # Core OpenVLA model
        self.openvla_model = OpenVLAModel(config)
        
        # Expert policy network for guidance
        self.expert_encoder = nn.Sequential(
            nn.Linear(config.max_state_dim, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        
        # Expert action embeddings (learned representations of expert actions)
        self.expert_action_embed = nn.Embedding(
            num_embeddings=config.chunk_size,
            embedding_dim=config.hidden_size
        )
        
        # Feature fusion layer for combining OpenVLA predictions with expert guidance
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Expert-guided prediction head
        self.expert_guided_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.max_action_dim)
        )
        
        # Expert confidence prediction network
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Confidence between 0 and 1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for expert integration modules."""
        for module in [self.expert_encoder, self.fusion_layer, self.expert_guided_head, self.confidence_head]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def encode_expert_state(self, expert_state: torch.Tensor) -> torch.Tensor:
        """Encode expert states into representation space."""
        masked_state = expert_state.clone()
        
        # Optionally mask expert states for robustness (like in BCZ)
        if self.training and self.config.get('expert_dropout', 0.0) > 0:
            mask = torch.rand_like(masked_state) > self.config.expert_dropout
            masked_state = masked_state * mask
        
        return self.expert_encoder(masked_state)
    
    def get_expert_guidance(self, expert_actions: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
        """Generate expert guidance embeddings from expert actions."""
        batch_size, seq_len = expert_actions.shape[:2]
        device = expert_actions.device
        
        # Create positional indices for expert action embeddings
        pos_indices = seq_positions.unsqueeze(-1).expand(batch_size, seq_len)
        
        # Embed expert actions
        expert_embed = self.expert_action_embed(pos_indices.mean(dim=1).long())
        
        # Create guidance signal based on expert actions
        expert_guidance = expert_embed.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        return expert_guidance
    
    def forward_with_expert(
        self,
        images: torch.Tensor,
        language_input: str,
        state: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        expert_states: torch.Tensor = None,
        expert_actions: torch.Tensor = None,
        expert_masks: torch.Tensor = None,
    ):
        """
        Forward pass with expert guidance.
        
        Args:
            images: [B, C, H, W] Vision input
            language_input: Natural language instruction
            state: [B, S] Current robot state
            actions: [B, T, A] Target actions
            timesteps: [B] Diffusion timesteps
            expert_states: [B, S'] Optional expert states
            expert_actions: [B, T, A] Optional expert actions
            expert_masks: [B, T] Mask for valid expert actions
            
        Returns:
            dict: Contains predictions, expert guidance, loss, and confidence scores
        """
        # Base OpenVLA predictions
        base_predictions = self.openvla_model(
            images=images,
            language_input=language_input,
            state=state,
            actions=actions,
            timesteps=timesteps,
        )
        
        # Expert guidance integration
        expert_guidance = None
        if expert_states is not None and expert_actions is not None:
            expert_states_encoded = self.encode_expert_state(expert_states)
            
            # Generate expert guidance
            seq_positions = torch.arange(base_predictions.shape[1], device=base_predictions.device)
            expert_guidance = self.get_expert_guidance(expert_actions, seq_positions)
            
            # Combine base predictions with expert guidance
            # Use cross-attention to fuse information
            expert_context = expert_states_encoded.unsqueeze(1).expand_as(expert_guidance)
            
            # Create query = base predictions, key/value = expert guidance
            fused_output, attention_weights = self.fusion_layer(
                query=base_predictions,
                key=expert_context,
                value=expert_guidance,
                key_padding_mask=~expert_masks.bool() if expert_masks is not None else None
            )
            
            # Generate final predictions with expert guidance
            combined_features = torch.cat([base_predictions, fused_output], dim=-1)
            final_predictions = self.expert_guided_head(combined_features)
        else:
            final_predictions = base_predictions
            attention_weights = None
        
        # Predict expert confidence
        confidence = self.confidence_head(final_predictions.mean(dim=1))
        
        return {
            'predictions': final_predictions,
            'base_predictions': base_predictions,
            'expert_guidance': expert_guidance,
            'confidence': confidence,
            'attention_weights': attention_weights,
        }
    
    def compute_expert_losses(self, outputs: dict, targets: torch.Tensor, expert_targets: torch.Tensor) -> dict:
        """
        Compute losses for expert-guided training.
        
        Args:
            outputs: Outputs from forward_with_expert
            targets: Ground truth target actions
            expert_targets: Expert demonstration actions
            
        Returns:
            dict: Various loss components
        """
        losses = {}
        
        # Primary action prediction loss
        predictions = outputs['predictions']
        losses['action_loss'] = nn.functional.mse_loss(predictions, targets)
        
        # Behavior cloning loss on expert data
        if expert_targets is not None:
            # Weight loss by confidence in expert data
            confidence_weight = outputs['confidence'].detach()
            bc_loss = nn.functional.mse_loss(predictions, expert_targets, reduction='none')
            bc_loss = bc_loss * confidence_weight.unsqueeze(-1).unsqueeze(-1)
            losses['bc_loss'] = bc_loss.mean()
        
        # Confidence regularization (encourage using expert knowledge)
        if self.training and expert_targets is not None:
            losses['confidence_regularization'] = (-outputs['confidence']).mean()
        
        # Attention regularization (smooth attention patterns)
        if outputs['attention_weights'] is not None:
            attention_reg = torch.var(outputs['attention_weights'], dim=-1).mean()
            losses['attention_regularization'] = 0.001 * attention_reg
        
        return losses
    
    def imitate_expert(self, expert_demo: dict, beta: float = 1.0) -> dict:
        """
        Implement behavioral cloning on expert demonstrations.
        
        Args:
            expert_demo: Dictionary containing expert data
            beta: Temperature for action sampling
            
        Returns:
            dict: Results from expert imitation
        """
        # Prepare expert data
        images = expert_demo.get('images')
        language = expert_demo.get('instruction', "")
        state = expert_demo.get('current_state')
        expert_actions = expert_demo.get('expert_actions')
        
        if any(x is None for x in [images, state, expert_actions]):
            raise ValueError("Missing required expert data fields")
        
        # Generate random timesteps for training
        batch_size = images.shape[0]
        device = images.device
        timesteps = torch.rand(batch_size, device=device)
        
        # Forward pass with expert guidance
        outputs = self.forward_with_expert(
            images=images,
            language_input=language,
            state=state,
            actions=expert_actions,  # Use expert actions as targets
            timesteps=timesteps,
            expert_states=state,
            expert_actions=expert_actions,
        )
        
        # Compute imitation losses
        losses = self.compute_expert_losses(outputs, expert_actions, expert_actions)
        
        # Combine losses
        total_loss = (
            losses.get('action_loss', 0) +
            losses.get('bc_loss', 0) +
            losses.get('confidence_regularization', 0) +
            losses.get('attention_regularization', 0)
        )
        
        losses['total_loss'] = total_loss
        
        return {
            **losses,
            'predictions': outputs['predictions'],
            'confidence': outputs['confidence'],
        }
    
    def validate_expert_data(self, expert_data: dict) -> bool:
        """Validate expert data integrity."""
        required_keys = ['images', 'current_state', 'expert_actions']
        return all(key in expert_data for key in required_keys)


class OpenVLAExpertTrainer:
    """
    Trainer class for expert-guided OpenVLA training.
    
    Handles expert data preprocessing, training loops, and evaluation.
    """
    
    def __init__(self, model: OpenVLAWithExpert, config: OpenVLAConfig):
        self.model = model
        self.config = config
    
    def prepare_expert_batch(self, expert_data: dict) -> dict:
        """Prepare expert data for training batch."""
        batch = {}
        
        # Validate required fields
        if not self.model.validate_expert_data(expert_data):
            raise ValueError("Invalid expert data format")
        
        # Basic preprocessing
        batch['images'] = expert_data['images'].float()
        batch['current_state'] = expert_data['current_state'].float()
        batch['expert_actions'] = expert_data['expert_actions'].float()
        
        # Add language instruction if provided
        if 'instruction' in expert_data:
            batch['instruction'] = expert_data['instruction']
        else:
            batch['instruction'] = ["Follow expert demonstration"] * len(expert_data['images'])
        
        # Create expert masks for valid actions
        if 'expert_masks' in expert_data:
            batch['expert_masks'] = expert_data['expert_masks'].bool()
        else:
            batch['expert_masks'] = torch.ones_like(expert_data['expert_actions'][:, :, 0], dtype=torch.bool)
        
        return batch
    
    def expert_epoch(self, expert_loader, optimizer, device='cuda'):
        """Run one epoch of expert training."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, expert_data in enumerate(expert_loader):
            # Move to device
            for key in expert_data:
                if isinstance(expert_data[key], torch.Tensor):
                    expert_data[key] = expert_data[key].to(device)
            
            # Prepare batch
            prepared_batch = self.prepare_expert_batch(expert_data)
            
            # Forward pass
            result = self.model.imitate_expert(prepared_batch)
            
            # Backward pass
            optimizer.zero_grad()
            result['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += result['total_loss'].item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
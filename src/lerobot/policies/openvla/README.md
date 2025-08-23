# OpenVLA Policy Implementation

This directory contains the OpenVLA (Visual-Language-Action) policy implementation for the LeRobot framework, specifically adapted for Unitree G1 robot integration.

## Overview

OpenVLA is a multimodal policy that combines:
- **Vision Understanding**: Processes camera images through a Vision-Language Model (VLM)
- **Language Understanding**: Interprets natural language instructions
- **Action Prediction**: Generates robot actions using a specialized action expert

## Architecture

### Core Components

1. **OpenVLAConfig**: Configuration class inheriting from `PreTrainedConfig`
2. **OpenVLAModel**: Main neural architecture combining VLM with action prediction
3. **OpenVLAActionExpert**: Specialized decoder for action generation
4. **OpenVLAPolicy**: LeRobot-compatible policy wrapper
5. **OpenVLAWithExpert**: Enhanced version with expert demonstration integration

### Key Features

- **VLM Integration**: Supports HuggingFace VLM models (default: SmolVLM2-500M)
- **Fallback Mode**: Gracefully falls back to learned embeddings if VLM loading fails
- **Action Chunking**: Supports temporal action sequences for smooth robot control
- **Expert Integration**: Can leverage expert demonstrations for improved training
- **LeRobot Compatibility**: Fully integrated with LeRobot training and evaluation framework

## Configuration

### Basic Configuration

```python
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig

config = OpenVLAConfig(
    vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    hidden_size=512,
    max_action_dim=32,
    max_state_dim=32,
    chunk_size=50,
    n_action_steps=50
)
```

### Key Parameters

- `vlm_model_name`: HuggingFace model identifier for VLM backbone
- `hidden_size`: Hidden dimension for action expert (should match VLM)
- `max_action_dim`: Maximum action dimension across supported robots
- `max_state_dim`: Maximum robot state dimension
- `chunk_size`: Number of action timesteps to predict
- `freeze_vision_encoder`: Whether to freeze VLM parameters during training

## Usage

### Training

```python
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy

# Create policy
policy = OpenVLAPolicy(config)

# Training loop
for batch in dataloader:
    loss = policy(batch)
    loss.backward()
    optimizer.step()
```

### Inference

```python
# Single action prediction
action = policy.select_action(observation_batch)

# Full action chunk prediction
actions = policy.sample_action_chunk(observation_batch)
```

## Integration with LeRobot

The OpenVLA policy is fully integrated with the LeRobot framework:

1. **Factory Registration**: Available via `get_policy_class("openvla")`
2. **Configuration Management**: Uses LeRobot's configuration system
3. **Normalization**: Integrates with LeRobot's data normalization
4. **Training Scripts**: Compatible with LeRobot training pipelines

## Dependencies

- `torch`: PyTorch for neural network operations
- `transformers`: HuggingFace transformers for VLM integration
- `lerobot`: LeRobot framework for policy management

Install with:
```bash
pip install torch transformers
```

## Recent Fixes

### Configuration Issues
- ✅ Fixed inheritance from `PolicyConfig` to `PreTrainedConfig`
- ✅ Added proper LeRobot configuration registration
- ✅ Aligned parameters with SmolVLA configuration structure

### Factory Integration
- ✅ Added OpenVLA to LeRobot policy factory
- ✅ Registered configuration creation function
- ✅ Fixed import paths and dependencies

### VLM Integration
- ✅ Replaced random tensor placeholders with actual VLM processing
- ✅ Added graceful fallback to learned embeddings
- ✅ Implemented proper image and language processing pipelines

### Code Quality
- ✅ Added dependency checks and error handling
- ✅ Fixed configuration validation
- ✅ Improved error messages and debugging

## Testing

Run the test script to verify the implementation:

```bash
cd lerobot/src/lerobot/policies/openvla
python test_openvla.py
```

## Examples

See the configuration files in:
- `config/eval/config_eval_openvla/` for evaluation setup
- `config/train/config_train_openvla/` for training setup

## Troubleshooting

### Common Issues

1. **VLM Loading Failed**: Check internet connection and model availability
2. **Configuration Errors**: Ensure all required parameters are set
3. **Import Errors**: Verify LeRobot installation and path setup

### Debug Mode

Enable debug logging to see detailed VLM processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When modifying the OpenVLA implementation:

1. Maintain compatibility with LeRobot framework
2. Test with both VLM and fallback modes
3. Update configuration validation as needed
4. Run the test suite before submitting changes

## License

This implementation follows the same license as the LeRobot framework.

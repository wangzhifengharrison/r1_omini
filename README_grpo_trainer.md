# Qwen2VL GRPO Trainer

This file implements the `Qwen2VLGRPOTrainer` class, which extends Hugging Face's `Trainer` class to support Group Relative Policy Optimization (GRPO) for vision-language models.

## Overview

The `Qwen2VLGRPOTrainer` is designed for training vision-language models with GRPO - an algorithm initially proposed in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300). This implementation specifically adapts GRPO for multimodal inputs including images and videos.

## Key Features

- Handles multimodal inputs (images and videos)
- Implements GRPO algorithm for vision-language models
- Supports reward-based model optimization
- Works with Qwen2-VL, Qwen2.5-VL, and Aria models
- Processes images and videos for multimodal understanding

## File Structure

```
R1-Omni/
├── src/
│   └── r1-v/
│       ├── src/
│       │   └── open_r1/
│       │       ├── grpo.py             # Base GRPO implementation
│       │       └── trainer/
│       │           ├── grpo_trainer.py            # This file
│       │           └── humanOmni_grpo_trainer.py  # Extension for HumanOmni
│       ├── run_grpo_humanomni.sh       # Training script
│       └── wandb/                      # Logging directory
├── humanomni/                          # HumanOmni model implementation
├── trl/                                # TRL library
│   └── trainer/
│       └── grpo_config.py              # GRPO configuration
└── qwen_vl_utils.py                    # Utilities for Qwen VL models
```

## Function Calling Relationship

```
Qwen2VLGRPOTrainer.__init__
    ├── Initialize model and configurations
    ├── Set up reference model (for KL divergence)
    ├── Configure processing classes
    └── Prepare reward functions

compute_loss
    ├── Process input prompts (text, image/video)
    ├── Generate completions using model.generate()
    ├── For images: _get_per_token_logps
    │   └── model() (Forward pass)
    ├── For videos: _get_per_token_logps_video
    │   └── model() (Forward pass)
    ├── Calculate rewards using reward_funcs
    ├── Compute advantages and KL divergence
    └── Calculate final loss

_get_per_token_logps / _get_per_token_logps_video
    ├── Run model forward pass
    └── Calculate per-token log probabilities

_prepare_inputs
    └── Skip standard preparation (overridden)

_set_signature_columns_if_needed
    └── Set columns expected by training_step

log
    ├── Average metrics
    └── parent.log() (Call parent class log)

create_model_card
    └── Generate model card README
```

## Usage

The trainer is typically used as part of a training script. Below is a simplified example:

```python
from src.r1-v.src.open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig

# Initialize trainer
trainer = Qwen2VLGRPOTrainer(
    model="Qwen/Qwen2-VL-2B-Instruct",
    reward_funcs="path/to/reward_model",
    args=GRPOConfig(output_dir="output"),
    train_dataset=dataset,
)

# Train the model
trainer.train()
```

## Implementation Details

The trainer performs the following key steps during training:

1. Processes multimodal inputs (images or videos with text)
2. Generates multiple completions for each input
3. Calculates rewards for each completion
4. Computes advantages by normalizing rewards
5. Applies GRPO loss calculation with KL penalty
6. Updates the model parameters

The implementation handles both image and video inputs, with specialized processing for each modality type. It maintains a reference model to calculate KL divergence, ensuring the model doesn't deviate too far from the initial distribution while optimizing for rewards. 
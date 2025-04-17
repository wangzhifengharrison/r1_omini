# HumanOmni GRPO Trainer

This file implements the `HumanOmniVLGRPOTrainer` class, which extends Hugging Face's `Trainer` class to support Group Relative Policy Optimization (GRPO) for multimodal models, particularly those handling video, audio, and text.

## Overview

The `HumanOmniVLGRPOTrainer` is designed for training multimodal models with GRPO - an algorithm initially proposed in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300). This implementation specifically adapts GRPO for multimodal inputs including video and audio.

## Key Features

- Handles multimodal inputs (video, audio, text)
- Implements GRPO algorithm for multimodal models
- Supports reward-based model optimization
- Works with Qwen2-VL and similar vision-language models
- Processes videos using SiglipVisionModel for visual features
- Processes audio using WhisperProcessor for audio features

## File Structure

```
R1-Omni/
├── humanomni/
│   ├── model/
│   │   ├── humanomni_arch.py     # Architecture definitions
│   │   ├── humanomni_model.py    # Model definitions
│   │   └── __pycache__/
│   ├── constants.py              # Constants like token definitions
│   └── mm_utils.py               # Multimodal utility functions
├── src/
│   └── r1-v/
│       ├── src/
│       │   └── open_r1/
│       │       ├── grpo.py       # Base GRPO implementation
│       │       └── trainer/
│       │           ├── humanOmni_grpo_trainer.py  # This file
│       │           └── grpo_trainer.py            # Base GRPO trainer
│       └── run_grpo_humanomni.sh                  # Training script
└── siglip-base-patch16-224/      # Vision model
└── whisper-large-v3/             # Audio model
```

## Function Calling Relationship

```
HumanOmniVLGRPOTrainer.__init__
    ├── Initialize models and configurations
    ├── Load vision and audio towers
    └── Prepare models with accelerator

compute_loss
    ├── Process input prompts (text, video, audio)
    ├── Generate completions using model.generate()
    ├── _get_per_token_logps_video (Get token log probabilities)
    │   └── model() (Forward pass)
    ├── Calculate rewards using reward_funcs
    ├── Compute advantages and KL divergence
    └── Calculate final loss

_get_per_token_logps_video
    ├── model() (Forward pass)
    └── Calculate per-token log probabilities

_prepare_inputs
    └── Skip standard preparation (overridden)

log
    ├── Average metrics
    └── parent.log() (Call parent class log)

create_model_card
    └── Generate model card README
```

## Usage

The trainer is typically used as part of a training script. Below is a simplified example:

```python
from src.r1-v.src.open_r1.trainer.humanOmni_grpo_trainer import HumanOmniVLGRPOTrainer
from trl import GRPOConfig

# Initialize trainer
trainer = HumanOmniVLGRPOTrainer(
    model="path/to/model",
    reward_funcs="path/to/reward_model",
    args=GRPOConfig(output_dir="output"),
    train_dataset=dataset,
)

# Train the model
trainer.train()
```

## Implementation Details

The trainer performs the following key steps during training:

1. Processes multimodal inputs (video, audio, text)
2. Generates multiple completions for each input
3. Calculates rewards for each completion
4. Computes advantages by normalizing rewards
5. Applies GRPO loss calculation with KL penalty
6. Updates the model parameters

The implementation maintains a reference model to calculate KL divergence, ensuring the model doesn't deviate too far from the initial distribution while optimizing for rewards. 
#!/bin/bash
# RunPod setup script - pre-existing python 3.12
set -e

echo "Setting up RunPod environment..."

# quick GPU check
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# dependencies
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "Dependencies installed"
else
    echo "ERROR: requirements.txt not found"
    exit 1
fi

python3 -c "import torch, transformers, peft" || {
    echo "ERROR: Missing critical packages"
    exit 1
}

# ?flash-attn
python3 -c "import flash_attn" 2>/dev/null || {
    echo "WARNING: flash-attn not installed. Install with:"
    echo "  pip install flash-attn --no-build-isolation"
}

echo ""
echo "Setup complete! Next steps:"
echo "  1. Ensure train.jsonl is present"
echo "  2. Review config.yaml settings"
echo "  3. Start training: python train.py"

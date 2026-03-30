#!/bin/bash

clear
rm -rf ./checkpoints/feature_routed/*
./venv/bin/python train.py

# Extract the checkpoint with the highest F1 score
BEST_CHECKPOINT=$(ls ./checkpoints/feature_routed/*.pt 2>/dev/null | awk -F'_f1' '{print $2 "\t" $0}' | sort -n | tail -n 1 | cut -f2)

# Check if a file was actually found
if [ -z "$BEST_CHECKPOINT" ]; then
    echo "Error: No checkpoints found in ./checkpoints/feature_routed/"
    exit 1
fi

echo "Training complete. Optimal weights saved at:"
echo "$BEST_CHECKPOINT"

./venv/bin/python verify.py --checkpoint "$BEST_CHECKPOINT"

# You can now automatically pass this variable to your visualizer
./venv/bin/python show.py --checkpoint "$BEST_CHECKPOINT"

./venv/bin/python trade.py --checkpoint "$BEST_CHECKPOINT"

./venv/bin/python strategy.py --checkpoint "$BEST_CHECKPOINT"
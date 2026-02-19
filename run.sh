#!/bin/bash

# Give execution permission: chmod +x run.sh

echo "Running Naive KD..."
python kd.py --batch_size 128 --epochs 100 --alpha 0.9 --temperature 4.0

echo "Running Representation Similarity Distillation..."
python rsd.py --batch_size 128 --epochs 100 --sp_weight 3000.0
#!/bin/bash

# Give execution permission: chmod +x run.sh

echo "Training the Teacher Model..."
python teacher.py --arch resnet18 --epochs 200 --device 0 --lr 0.1

echo "Training the Student Model..." 
python teacher.py --student --epochs 100 --device 0 --lr 0.1 

echo "Running Naive KD..."
python kd.py --batch_size 128 --epochs 100 --alpha 0.9 --temperature 4.0 --device cuda:0 --teacher_path saved_model/teacher_resnet18_best.pth

echo "Running Representation Similarity Distillation..."
python spd.py --batch_size 128 --epochs 100 --sp_weight 3000.0 --device cuda:0 --teacher_path saved_model/teacher_resnet18_best.pth
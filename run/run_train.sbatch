#!/bin/bash
#SBATCH --job-name=temp_swinir_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=128G
#SBATCH --time=72:00:00

# Для распределенного обучения на 2 GPU (если нужно)
# #SBATCH --gres=gpu:a100:2
# #SBATCH --ntasks-per-node=2

echo "============================================"
echo "Temperature SwinIR Training Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Настройка окружения
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Директория проекта
PROJECT_DIR=$HOME/temperature_swinir
cd $PROJECT_DIR

# Скачивание предобученной модели если еще нет
PRETRAINED_DIR=$PROJECT_DIR/pretrained
mkdir -p $PRETRAINED_DIR

if [ ! -f "$PRETRAINED_DIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth" ]; then
    echo "Downloading pretrained model..."
    wget -P $PRETRAINED_DIR https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth
fi

# Установка зависимостей
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user torch torchvision torchaudio timm opencv-python tqdm tensorboard matplotlib scipy pyyaml

# Проверка окружения
echo "Verifying environment..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
import cv2
print(f'OpenCV: {cv2.__version__}')
import timm
print(f'Timm: {timm.__version__}')
import scipy
print(f'SciPy: {scipy.__version__}')
"

echo "============================================"
echo "Starting Temperature SwinIR training..."

# Определение пути к данным (остается прежним)
DATA_DIR=$HOME/temperature_sr_project/data

# Запуск обучения
# Для одной GPU:
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $DATA_DIR:$DATA_DIR \
    --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python train.py

# Для распределенного обучения на 2 GPU (раскомментировать если нужно):
# export MASTER_PORT=29500
# export WORLD_SIZE=2
# apptainer exec --nv \
#     --bind $HOME/local-python:$HOME/.local \
#     --bind $PROJECT_DIR:$PROJECT_DIR \
#     --bind $DATA_DIR:$DATA_DIR \
#     --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
#     /home/shared/containers/tensorflow-25.02-py3.sif \
#     torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
#     train.py

echo "============================================"
echo "Temperature SwinIR Training Job Finished: $(date)"
echo "============================================"
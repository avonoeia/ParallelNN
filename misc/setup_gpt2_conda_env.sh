#!/bin/bash

# Define environment name
env_name="gpt2"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create a new conda environment
conda create -y --name $env_name python=3.9

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $env_name

# Install PyTorch with CUDA support
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify installation
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Install additional dependencies
pip install torch torchvision torchaudio transformers accelerate datasets

echo "Conda environment '$env_name' has been set up successfully!"

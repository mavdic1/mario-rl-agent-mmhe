#!/bin/bash

# Initialize conda for script usage
source $(conda info --base)/etc/profile.d/conda.sh

# Create environment with specified python version
conda create -n mario python=3.8 -y

# Enable the new environment
conda activate mario

# Force specific pip version for consistent dependency resolution
python -m pip install pip==23.3.1

# Install emulator and environment interface
pip install gym==0.21.0
pip install gym-retro==0.8.0

# Install core reinforcement learning libraries
pip install stable-baselines3==1.8.0
pip install torch==2.4.1

# Install data handling and math libraries
pip install numpy==1.24.4
pip install pandas==2.0.3

# Install vision processing and progress tracking
pip install opencv-python==4.13.0.92
pip install tqdm==4.67.3

# Install logging tools
pip install tensorboard==2.14.0

echo "Setup complete"
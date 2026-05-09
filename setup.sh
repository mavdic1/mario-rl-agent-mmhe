#!/bin/bash

echo "============================================"
echo "Mario PPO Setup (Python 3.8.20 required)"
echo "============================================"

# Verify Python version
python --version

echo "Installing pip tooling..."

python -m pip install "pip<24.1"

python -m pip install \
    "pip==24.0" \
    "setuptools==65.5.0" \
    "wheel==0.38.4" \
    "packaging==21.3" \
    --force-reinstall

echo "Installing core ML stack..."

pip install gym==0.21

pip install stable-baselines3==1.8.0

pip install gym-retro==0.8.0

pip install opencv-python==4.13.0.92

echo "Installing project requirements..."

pip install -r requirements.txt

echo "============================================"
echo "Setup complete!"
echo "============================================"
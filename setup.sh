#!/bin/bash

# create conda env
conda create -n mario python=3.8 -y
conda activate mario

# install dependencies
pip install -r requirements.txt

# import retro ROMs
python -m retro.import .

echo "Setup complete"
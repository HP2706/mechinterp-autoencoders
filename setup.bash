#!/bin/bash
#NOTE not secure
conda init
source ~/.bashrc  # Ensure the shell is reloaded with conda initialized
conda create -n myenv -y python=3.11 
source ~/.bashrc
conda activate myenv
pip install uv
uv pip install -r requirements.txt
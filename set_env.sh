#!/bin/bash

# Set the PYTHONPATH environment variable
export PYTHONPATH=$PYTHONPATH:./recognize-anything
export PYTHONPATH=$PYTHONPATH:./GroundingDINO
export PYTHONPATH=$PYTHONPATH:./segment_anything

# Set CUDA environment variable
export CUDA_HOME=/usr/local/cuda-12.4
alias python=python3

# Set build options
export BUILD_WITH_CUDA=True
export AM_I_DOCKER=False
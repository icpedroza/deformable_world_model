
# ESE 546 Final Project: Learning and Planning within a Deformable World Model

## Important Citation
This codebase was built off of the original environment codebase created in:
```
@article{shi2022robocraft,
  title={RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Objects with Graph Networks},
  author={Shi, Haochen and Xu, Huazhe and Huang, Zhiao and Li, Yunzhu and Wu, Jiajun},
  journal={arXiv preprint arXiv:2205.02909},
  year={2022}
}
```
The original files for the class are:
- train.py 
- model.py
- dino_patch.py
- control.py

Links to the data and checkpoints are included in a separate Google Drive Folder

## Overview

This is the codebase of my 546 Project in the Plasticine Lab simulator.

## Prerequisites
- Linux or macOS (Tested on Ubuntu 20.04)
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Conda

## Getting Started

### Setup
```bash
# clone the repo
cd RoboCraft

# create the conda environment
conda env create -f robocraft.yml
conda activate robocraft

# install requirements for the simulator
cd simulator
pip install -e .
```

### Data Generation
- We ran all the blocks in `simulator/plb/algorithms/test_tasks.ipynb` to generate data. This was a modified script to generate the correct format of data for my model. It is easier to use ipython notebook when dealing with Taichi env for fast materialization. 


## Code structure
- The simulator folder contains the simulation environment we used for data collection and particle sampling. 
- The robocraft folder contains the code for learning the model and planning within it


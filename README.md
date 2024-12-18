# Topology Optimization with Physics-informed Gaussian Processes

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for the paper [Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes
](https://arxiv.org/abs/2408.03490), where we develop a new class of TO methods based on the framework of Gaussian processes (GPs) whose mean functions are parameterized via deep neural networks. 

## Overview
We place GP priors on all design and state variables to represent them via parameterized continuous functions. These GPs share a deep neural network as their mean function but have as many independent kernels as there are state and design variables. We estimate all the parameters of our model in a single for loop that optimizes a penalized version of the performance metric where the penalty terms correspond to the state equations and design constraints.

![Flowchart](https://github.com/Bostanabad-Research-Group/GP-for-TO/blob/main/images/flowchart.png?raw=true)


## Requirements
To set up the environment for this project, follow these steps:

1. **Create a new virtual environment**:  
   ```bash
   python -m venv GP_for_TO
   ```
   Activate the environment:  
   - On Windows:  
     ```bash
     GP_for_TO\Scripts\activate
     ```
   - On macOS/Linux:  
     ```bash
     source GP_for_TO/bin/activate
     ```

2. **Install the required packages**:
   - [PyTorch](https://github.com/pytorch/pytorch) == 1.12.0 & CUDA >= 11.3:  
     ```bash
     pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --index-url https://download.pytorch.org/whl/cu113
     ```
   - [GPyTorch](https://github.com/cornellius-gp/gpytorch) == 1.7.0:  
     ```bash
     pip install gpytorch==1.7.0
     ```
   - **Dill** == 0.3.5.1:  
     ```bash
     pip install dill==0.3.5.1
     ```
   - **pandas**:  
     ```bash
     pip install pandas
     ```
   - **Matplotlib** == 3.5.3:  
     ```bash
     pip install matplotlib==3.5.3
     ```
   - **Tqdm** >= 4.66.4:  
     ```bash
     pip install tqdm>=4.66.4
     ```

Once these steps are completed, your environment will be ready to use.

## Usage
Once you have downloaded the code from this GitHub repository and installed the required packages, you’re ready to get started. The primary script, main_TO.py, located in the notebook folder, demonstrates the application of our technique to the topology optimization (TO) problems detailed in the paper.


## Citing Us
If you use this code or find our work interesting, please cite the following paper:

Yousefpour, Amin, et al. "Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes." arXiv preprint arXiv:2408.03490 (2024).

## Assistance and Support
Need help with the code? Feel free to open an issue on our GitHub page and label it according to the module or feature in question for quicker assistance.




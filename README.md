# Conditional_Diffusion

Conditional Diffusion Model for Image Generation

This repository contains the code for training a class-conditional diffusion model for image generation, built using PyTorch. The project uses a powerful U-Net architecture with self-attention, based on the model from the DiffPIR/deepinv library, and is configured for training on a custom image dataset (e.g., Food-15).

The training pipeline is robust, featuring modern deep learning practices and full experiment tracking with Weights & Biases.

(A collage showing image quality progression from early to late epochs)

Key Features

This implementation is built for stability, performance, and clear experiment tracking.

Advanced Model Architecture: Utilizes a U-Net with self-attention layers (DiffUNet from DiffPIR), which is crucial for learning global image structures and generating coherent objects.

Class-Conditional Generation: The model is trained with class labels, allowing you to specify what kind of image to generate during inference (e.g., "pizza," "sushi").

Classifier-Free Guidance (CFG): Implements CFG during training and sampling, a key technique for improving sample quality and adherence to the class condition.

Robust Training Pipeline:

Exponential Moving Average (EMA): Maintains an EMA of the model's weights for more stable and higher-quality final results.

Gradient Accumulation: Allows for large virtual batch sizes to stabilize training, even on GPUs with limited VRAM.

Learning Rate Scheduling: Uses a linear warmup phase followed by a cosine annealing schedule to optimize convergence.

Mixed Precision: Utilizes torch.amp where compatible to speed up training and reduce memory usage (currently disabled for DiffUNet compatibility).

Experiment Tracking: Fully integrated with Weights & Biases (wandb) to log hyperparameters, track metrics (loss, LR), and visualize generated image samples in real-time from anywhere.

Flexible & Resilient:

Supports training at multiple resolutions (e.g., 64px, 128px, 256px).

Includes robust checkpointing to save both the latest and the best-performing model weights.

Supports resuming training from any saved checkpoint.

Setup and Installation

Follow these steps to set up the environment and run the code.

1. Clone the Repository

git clone <your-repository-url>
cd <your-repository-name>


2. Create a Conda Environment

It is highly recommended to use a Conda environment to manage dependencies.

conda create -n diffusion python=3.11
conda activate diffusion


3. Install Dependencies

Install PyTorch with CUDA support by following the official instructions at pytorch.org. Then, install the other required packages.

# Example for a specific CUDA version - check the PyTorch website for the latest command
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install other packages
pip install numpy Pillow tqdm wandb


4. Log in to Weights & Biases

You will need a free wandb account.

wandb login
# Follow the prompts and paste your API key


Dataset Preparation

The script expects a standard image folder structure where each class has its own subdirectory.

Food_Dataset/
├── class_01_pizza/
│   ├── 001.jpg
│   ├── 002.png
│   └── ...
├── class_02_sushi/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── class_15_.../
    ├── ...
    └── ...


Update the --dataset_path argument to point to the root Food_Dataset folder.

How to Use

The project consists of two main scripts: one for training and one for sampling.

1. Training the Model

The main training script is conditional_diffusion_v4.1_wandb.py. All hyperparameters can be adjusted via command-line arguments.

Example: Start a new training run at 256x256 resolution

python conditional_diffusion_v4.1_wandb.py ^
    --img_size 256 ^
    --batch_size 4 ^
    --accum_steps 16 ^
    --lr 1e-4 ^
    --total_epochs 400


(Note: ^ is the line continuation character for Windows Command Prompt. Use \ for Linux/macOS.)

The training progress, metrics, and generated images will be streamed to your wandb project dashboard.

Example: Resume training from a checkpoint

python conditional_diffusion_v4.1_wandb.py --resume_from "Outputs/latest_checkpoint.pth"


Example: Fine-tune a trained model with a lower learning rate

python conditional_diffusion_v4.1_wandb.py ^
    --resume_from "Outputs/best_checkpoint.pth" ^
    --lr 1e-5 ^
    --total_epochs 550


(Remember to comment out the scheduler.load_state_dict line in the script for fine-tuning to reset the scheduler)

2. Generating Images (Sampling)

Use the sample.py script to generate images from a trained model checkpoint.

Example: Generate 4 images of class 5

python sample.py ^
    --checkpoint "Outputs/best_checkpoint.pth" ^
    --label 5 ^
    --n_samples 4 ^
    --img_size 256


This will save a grid of 4 images named generated_label_5_cfg_7.5.png in your project folder.

Experiment with --cfg_scale:

A higher scale (e.g., 10.0) makes the model adhere more strictly to the class features.

A lower scale (e.g., 3.0) allows for more creative and diverse outputs.

File Structure

conditional_diffusion_v4.1_wandb.py: The main script for training the model.

diffusion_model.py: A self-contained module defining the DiffUNet architecture and all its dependencies.

sample.py: A standalone script for generating images from a saved checkpoint.

README.md: This file.

Acknowledgments

The DiffUNet model architecture and its helper modules are adapted from the official implementation of the paper "DiffPIR: Diffusion Model for Image Prior".

Original Paper: [suspicious link removed]

Original Repository: https://github.com/yuanzhi-zhu/DiffPIR

This project repackages the model into a self-contained module and integrates it into a custom, class-conditional training pipeline with Weights & Biases for educational and experimental purposes.

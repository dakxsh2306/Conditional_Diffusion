Conditional Diffusion Model for Image Generation

This repository contains the code for training a class-conditional diffusion model for image generation, built using PyTorch. The project uses a powerful U-Net architecture with self-attention, based on the model from the DiffPIR/deepinv library, and is configured for training on a custom image dataset.

The training pipeline is robust, featuring modern deep learning practices and full experiment tracking with Weights & Biases.

(A collage showing final, high-quality generated images)

Table of Contents

Key Features

Setup and Installation

Dataset Preparation

How to Use

Training the Model

Generating Images (Sampling)

Results

File Structure

Acknowledgments

Key Features

Advanced Model Architecture: Utilizes a U-Net with self-attention layers (DiffUNet from DiffPIR), crucial for learning global image structures and generating coherent objects.

Class-Conditional Generation: Trained with class labels, allowing you to specify what kind of image to generate during inference (e.g., "pizza," "sushi").

Classifier-Free Guidance (CFG): Implements CFG during training and sampling to improve sample quality and adherence to the class condition.

Robust Training Pipeline:

Exponential Moving Average (EMA): Maintains an EMA of model weights for more stable, higher-quality final results.

Gradient Accumulation: Allows for large virtual batch sizes, even on GPUs with limited VRAM.

Learning Rate Scheduling: Uses a linear warmup phase followed by a cosine annealing schedule.

Experiment Tracking: Fully integrated with Weights & Biases (wandb) to log hyperparameters, track metrics (loss, LR), and visualize generated image samples in real-time.

Flexible & Resilient:

Supports training at multiple resolutions (e.g., 64px, 128px, 256px).

Includes robust checkpointing to save both the latest and the best-performing models.

Supports resuming training from any saved checkpoint.

Setup and Installation

Follow these steps to set up the environment and run the code.

1. Clone the Repository

git clone <your-repository-url>
cd <your-repository-name>


2. Create a Conda Environment

conda create -n diffusion python=3.11
conda activate diffusion


3. Install Dependencies

Install PyTorch with CUDA support from the official website. Then, install the other required packages.

# Example for a specific CUDA version - check the PyTorch website for the latest command
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install other packages
pip install numpy Pillow tqdm wandb


4. Log in to Weights & Biases

You will need a free wandb account.

wandb login
# Follow the prompts and paste your API key from your wandb.ai/settings page


Dataset Preparation

The dataset used for this project is the Food-101 Dataset, from which a subset of 15 classes was selected.

The script expects a standard image folder structure where each class has its own subdirectory. Update the --dataset_path argument to point to the root of this folder.

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


How to Use

The project consists of two main scripts: one for training and one for sampling.

1. Training the Model

The main training script is conditional_diffusion_v4.1_wandb.py. All hyperparameters can be adjusted via command-line arguments.

▶️ Start a new training run at 256x256 resolution:

# Note: ^ is for Windows Command Prompt. Use \ for Linux/macOS.
python conditional_diffusion_v4.1_wandb.py ^
    --img_size 256 ^
    --batch_size 4 ^
    --accum_steps 16 ^
    --lr 1e-4 ^
    --total_epochs 400


The training progress, metrics, and generated images will be streamed to your wandb project dashboard.

🔄 Resume training from a checkpoint:

python conditional_diffusion_v4.1_wandb.py --resume_from "Outputs/latest_checkpoint.pth"


⚙️ Fine-tune a trained model with a lower learning rate:
(Remember to comment out the scheduler.load_state_dict line in the script to reset the scheduler)

python conditional_diffusion_v4.1_wandb.py ^
    --resume_from "Outputs/best_checkpoint.pth" ^
    --lr 1e-5 ^
    --total_epochs 550


2. Generating Images (Sampling)

Use the sample.py script to generate images from a trained model checkpoint.

▶️ Generate 4 images of class 5:

python sample.py ^
    --checkpoint "Outputs/best_checkpoint.pth" ^
    --label 5 ^
    --n_samples 4 ^
    --img_size 256


This will save a grid of 4 images named generated_label_5_cfg_7.5.png.

🎨 Experiment with --cfg_scale:

A higher scale (e.g., 10.0) makes the model adhere more strictly to the class features.

A lower scale (e.g., 3.0) allows for more creative and diverse outputs.

Results

Here is a comparison of image quality at different stages of training on a 128x128 resolution.

Epoch 14

Epoch 60

Epoch 500 (Final)







Initial object structures begin to form.

Textures become more defined and realistic.

Final abstract textures are highly detailed.

File Structure

conditional_diffusion_v4.1_wandb.py: The main script for training the model.

diffusion_model.py: A self-contained module defining the DiffUNet architecture and its dependencies.

sample.py: A standalone script for generating images from a saved checkpoint.

README.md: This file.

Acknowledgments

The DiffUNet model architecture and its helper modules are adapted from the official implementation of the paper "DiffPIR: Diffusion Model for Image Prior".

Original Paper: Zhu, Y., Liu, Y., & Wang, L. (2023). DiffPIR...

Original Repository: https://github.com/yuanzhi-zhu/DiffPIR

This project repackages the model into a self-contained module and integrates it into a custom, class-conditional training pipeline with Weights & Biases for educational and experimental purposes.

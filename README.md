# Deep Learning Final Project – Fall 2025

Author: Isidora Erakovic
Course: Deep Learning
Semester: Fall 2025

## Project Description

This project contains the full implementation of the Deep Learning Final Project.
All tasks are implemented in PyTorch using ResNet-18 as the main backbone.

The project explores different training strategies, loss functions, attention mechanisms,
and data augmentation using a Variational Autoencoder (VAE).
Model performance is evaluated using a Kaggle competition (onsite test set).

## Environment

Python version: 3.11
Framework: PyTorch

Required libraries:
- torch
- torchvision
- numpy
- pandas
- pillow
- scikit-learn

## Setup Instructions

1) Create a virtual environment

python -m venv .venv

2) Activate the virtual environment

Windows (PowerShell):
.venv\Scripts\activate

Linux / macOS:
source .venv/bin/activate

3) Install dependencies

pip install torch torchvision numpy pandas pillow scikit-learn

## Project Structure

final_project_resources/
- isidora.py
- README.txt
- code_template.py

Scripts:
- task1_1_make_onsite_submission.py
- task1_2_frozen_backbone_train_head.py
- task1_3_full_finetune.py
- task2_1_focal_loss.py
- task2_2_class_balanced_full_finetune.py
- task3_1_se_full_finetune.py
- task3_2_mha_full_finetune.py
- task4_vae_augment_and_finetune.py

Datasets:
- train.csv
- val.csv
- offsite_test.csv
- onsite_test_submission.csv
- train_aug.csv

Trained models (checkpoints/):
- isidora_task1-2.pt
- isidora_task2-2.pt
- isidora_task3-2.pt
- isidora_task4.pt

## Running the Project

Run a single task:
python isidora.py --task task3-2

Run all tasks:
python isidora.py --task all

## Kaggle Submission

Submission CSV format:
id, D, G, A

All submissions were successfully uploaded and evaluated.

## Notes

- ResNet-18 is used for all tasks for fair comparison.
- Attention-based models achieved the highest performance.
- VAE augmentation increased dataset size but did not outperform attention mechanisms.

Author: Isidora Erakovic

# Self-Learning-VAE-for-EEG-Artifact-Removal

CS-LWR: Self-Learning VAE for EEG Artifact Removal

This repository contains the implementation, benchmarks, and supporting tools for my MSc dissertation project: EEG artifact removal using deep learning with Variational Autoencoders (VAE) and self-learning strategies.

# Repository Structure

Benchmark A – Initial benchmark experiments with traditional methods and DAE baselines.

Benchmark B – Extended benchmarks with representative deep learning models (e.g., GRU-based, CNN-Transformer, mobile-DAE).

Me_VAE/Code – Customized VAE implementation in MATLAB and Python, including encoder–decoder structures, sampling layer, and training loop.

Self-learning – Explorations on online and self-learning strategies for adaptive EEG denoising.

Toolbox – Utility scripts for data preprocessing, segmentation, normalization, and evaluation metrics (RRMSE, CC, PSD, etc.).

LICENSE – Repository license.

README.md – Project documentation.

# Project Overview

The goal of this project is to remove artifacts (EOG, EMG, motion) from EEG recordings using modern deep learning architectures. Building on conventional autoencoder designs, this work extends to:

Variational Autoencoder (VAE) with customized sampling layers.

Hybrid modeling combining CNN, RNN, and Transformer components.

Self-learning framework for adaptive online EEG denoising without strict supervision.

# Features

Full VAE and Autoencoder implementations (MATLAB + Python).

Custom training pipeline with minibatch queue, Adam optimizer, and reconstruction + KL loss.

Evaluation metrics in time and frequency domains (RMSE, RRMSE, Pearson CC, PSD-RRMSE).

Scalogram-based visualization for qualitative assessment.

Online self-learning experiments for real-time adaptability.

# Benchmarks

Two levels of benchmarks are included:

Benchmark A: Classical signal processing baselines (band-pass filtering, notch filtering, DAE).

Benchmark B: Comparison with state-of-the-art deep learning models for EEG denoising.

# Status

Ongoing development. Current version includes multiple submissions (submit, the-second-submit, the-fourth-submit, sixth submit) that reflect progressive experiments. It is notable that these documents are not the whole story, just for fast verification and reproduction. If you are interested in the training process, please email at ShiCheng056@outlook.com.

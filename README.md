# Self-Learning-VAE-for-EEG-Artifact-Removal

CS-LWR: Self-Learning VAE for EEG Artifact Removal

This repository contains the implementation, benchmarks, and supporting tools for my MSc dissertation project: Self-learning Variational Autoencoder for EEG Artifact Removal.

# Repository Structure

Benchmark A – Initial benchmark experiments with traditional filtering methods and DL baselines.

Benchmark B – Extended benchmarks with representative deep learning models (mobile-DAE).

Me_VAE/Code – Customized VAE implementation in MATLAB and Python, including encoder–decoder structures and training loop. It is notable that 2D convolutional layer with specific filter setting [heinght:1, width:3] is applied to support customized training, not normal 1D convolutional layer, but the effect is the same as you can monitor the training process by pause and check in workspace.

Self-learning – Explorations on online and self-learning strategies for adaptive EEG denoising.

Toolbox – Utility scripts for data preprocessing, segmentation, normalization, and evaluation metrics (RRMSE, CC, PSD, etc.).

LICENSE – Repository license.

README.md – Project documentation.

# Project Overview

The goal of this project is to remove artifacts (EOG, EMG, motion) from EEG recordings using modern deep learning architectures. Building on conventional autoencoder designs, this work extends to:

Variational Autoencoder (VAE) with customized architecture.

Hybrid modeling combining CNN, RNN, and Transformer components.

Self-learning framework for adaptive online EEG denoising without supervision.

# Features

DAE and DL models implementation (MATLAB, Python).

Custom training pipeline with Me-VAE architecture.

Evaluation metrics in time and frequency domains (RMSE, RRMSE, Pearson CC, PSD-RRMSE).

Visualization for qualitative assessment.

Online self-learning experiments for real-time adaptability (simplified by pure MATLAB implementation without Simulink).

# Benchmarks

Two levels of benchmarks are included:

Benchmark A: Classical signal processing baselines (band-pass filtering, DL models).

Benchmark B: Comparison with state-of-the-art deep learning models for EEG denoising (mobile DAE).

# Dataset of Me-VAE and Self-learning Part

Due to the 1G restriction of Git repository, part of dataset document and history record is included in Link: https://pan.baidu.com/s/1OsebeGXQVAElKzR18QE6aA (code: t9ic) 

# Status

Ongoing development. Current version includes multiple submissions (submit, the-second-submit, the-fourth-submit, sixth submit) that reflect progressive experiments. It is notable that these documents are not the whole story, just for fast verification and reproduction. If you are interested in the training process or other details, please email at ShiCheng056@outlook.com.

# Citation
A) Benchmark A:

  [1] H. Zhang, M. Zhao, C. Wei, D. Mantini, Z. Li and Q. Liu, "EEGdenoiseNet: A Benchmark Dataset for Deep Learning Solutions of EEG Denoising," Journal of Neural Engineering, vol. 18, no. 5, p. 056057, Oct. 2021, doi: 10.1088/1741-2552/ac2bf8.
Keywords (from the paper): deep learning; neural network; EEG dataset; benchmark dataset; EEG artifact removal; EEG denoising. Git Link: https://github.com/ncclabsustech/EEGdenoiseNet

B) Benchmark B:

  [2] L. Xing and A. J. Casson, "Deep Autoencoder for Real-Time Single-Channel EEG Cleaning and Its Smartphone Implementation Using TensorFlow Lite With Hardware/Software Acceleration," in IEEE Transactions on Biomedical Engineering, vol. 71, no. 11, pp. 3111-3122, Nov. 2024, doi: 10.1109/TBME.2024.3408331.
keywords: Electroencephalography;Signal processing algorithms;Brain modeling;Real-time systems;Electromyography;Electrooculography;Deep learning;Autoencoders;Smart phones;Tensors;Deep learning;convolutional autoencoder;EEG artifact removal;smartphone;tensorflow lite. Git Link: https://github.com/Non-Invasive-Bioelectronics-Lab/Autoencoder.git

C) VAE support function: 
Deeplearning toolbox function (modified in this work):

  [3] MathWorks, “Train Variational Autoencoder (VAE) to Generate Images,” MATLAB Documentation (Deep Learning Toolbox), 2025. [Online]. Available: https://uk.mathworks.com/help/deeplearning/ug/train-a-variational-autoencoder-vae-to-generate-images.html. [Accessed: 2-Sep-2025].

  [4] Diederik P. Kingma and Max Welling (2019), "An Introduction to Variational Autoencoders", Foundations and Trends® in Machine Learning: Vol. 12: No. 4, pp 307-392. http://dx.doi.org/10.1561/2200000056 

  The toolbox strictly follows the original VAE principle which is independent of any specific programming environment. While our MATLAB implementation (supplementary only for fast validation) employs modified support functions (sampling layer, projection layer, batch function) adapted from the Deep Learning Toolbox example [3], the same mechanism can equivalently be realized in Python or other deep learning frameworks. Thus, [3] is cited as a reference for implementation support only in this github implementation, while the theoretical foundation is attributed to [4].

D) Format transform function (included in the "Toobox"):

  [5] kwikteam, “npy-matlab: Read/Write NPY files in MATLAB,” GitHub repository, 2017. [Online]. Available: https://github.com/kwikteam/npy-matlab. [Accessed: 2-Sep-2025].

  The npy-matlab toolbox [3] was employed solely in the verification materials to support flexible interoperability between MATLAB and Python environments. This utility served only as a data handling aid and is not part of the methodological evaluation or contribution of this work.




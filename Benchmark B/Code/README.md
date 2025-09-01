Folder Overview

This folder documents how the autoencoder model was trained and evaluated.

Autoencoder Training

1. BigModel_v4.py is the end-to-end script for the autoencoder workflow. It includes:

2. EEG data loading

3. Segmentation (configurable; default 4 s windows with 50% overlap)

4. Per-segment min–max normalization

5. Dataset assembly for EOG / Motion / EMG, keeping ground truth and contaminated pairs aligned in the same order

6. Train/Validation/Test split

7. Model training

8. Model evaluation (time-domain and frequency-domain metrics)

9. Utilities: save/load Keras models, TFLite conversion, result caching, and plotting


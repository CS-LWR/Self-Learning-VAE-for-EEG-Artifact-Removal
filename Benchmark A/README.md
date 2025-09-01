Benchmark A

This code provides the implementation of Benchmark A, which serves as a comparative evaluation framework for different neural network architectures in EEG artifact removal.
It allows training and testing of multiple denoising models (e.g., fcNN, CNN, RNN-LSTM) under controlled experimental settings.

The benchmark is based on the original work of Haoming Zhang, the author of EEGdenoiseNet, who first released this dataset and framework for standardized comparison in EEG denoising research. It is then modified by Shi Cheng to make the workflow easier to reproduce, customize, and extend for further EEG denoising research.

Researchers can adjust parameters such as epochs, batch size, optimizer, and noise type (EOG/EMG) to reproduce or extend the benchmark experiments.
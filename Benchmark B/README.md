Benchmark B

This code provides Benchmark B comparisons for EEG artifact removal, enabling side-by-side evaluation of multiple denoising models under a unified protocol (4-s segments with 50% overlap, per-segment min–max normalization, and metrics including RRMSE-time, RRMSE-frequency via Welch PSD, and Pearson CC). The setup mirrors the settings described in the original DAE study. 
 
Original method and reference:
Le Xing and Alexander J. Casson, Deep Autoencoder for Real-Time Single-Channel EEG Cleaning and Its Smartphone Implementation Using TensorFlow Lite With Hardware/Software Acceleration, IEEE Transactions on Biomedical Engineering, 71(11), 2024. The work targets single-channel EEG cleaning (EOG, motion, EMG) and demonstrates real-time performance on smartphones via TensorFlow Lite. Please cite this paper if you use these Benchmark B settings. 

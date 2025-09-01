Datasets in This Benchmark

The data here are the preprocessed EOG/Motion/EMG datasets prior to segmentation and normalization (i.e., ready to feed into the training pipeline where segmentation/normalization are applied).

Preprocessing Summary

EOG artifacts:

1. EEG_clean_EOG_bp: raw → band-pass 1–50 Hz → trim first and last 2 s
2. EEG_noisy_EOG_bp: raw → no filtering → trim first and last 2 s

Motion artifacts:

1. EEG_clean_motion_bp: raw → detrend → downsample to 200 Hz → band-pass 1–50 Hz → trim first and last 5 s
2. EEG_noisy_motion_bp: raw → detrend → downsample to 200 Hz → no filtering → trim first and last 5 s

EMG artifacts:

1. EEG_clean_EMG_bp: raw → band-pass 1–50 Hz
2. EEG_noisy_EMG_bp: raw → band-pass 1–50 Hz

Note: “bp” denotes band-pass filtered signals. Subsequent segmentation and normalization are performed within the training script, not here.
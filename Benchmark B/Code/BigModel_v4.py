# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:02:28 2023
@author: m29244lx
Modified on Tue August 28 16:15:31 2025 by Shi Cheng

Script: End-to-end pipeline for EEG denoising experiments (EOG / Motion / EMG).
- Loads preprocessed datasets (200 Hz; saved in .npy with potential format differences).
- Computes SNR statistics (for EMG set shown), optional filtering, segmentation,
  per-segment min–max normalization, dataset splits, and data integration.
- Trains a simple 1D convolutional autoencoder baseline and evaluates with:
  RRMSE (time), RRMSE (frequency via Welch PSD), and Pearson correlation (CC).
- Exports artifacts, plots (learning curves, examples, boxplots), and TFLite model.

Note:
* Only comments and section headers were rewritten for clarity; code logic is unchanged.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import math
import scipy.io

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import os


# %% Load preprocessed data
# All datasets are sampled at 200 Hz. Saved formats may vary across files.
EEG_clean_EOG  = np.load('EEG_clean_EOG_bp.npy',  allow_pickle=True)
EEG_noisy_EOG  = np.load('EEG_noisy_EOG_bp.npy',  allow_pickle=True)

# Motion data preprocessed in MATLAB (transpose + squeeze to match shape)
EEG_clean_motion = np.squeeze(np.transpose(np.load('EEG_clean_motion_bp.npy', allow_pickle=True)))
EEG_noisy_motion = np.squeeze(np.transpose(np.load('EEG_noisy_motion_bp.npy', allow_pickle=True)))

# EMG sets with higher SNR (lower-SNR example kept for reference)
# EEG_clean_EMG = np.load('EEG_clean_EMG.npy')
# EEG_noisy_EMG = np.load('EEG_noisy_EMG.npy')
EEG_clean_EMG = np.load('EEG_clean_EMG_bp.npy')
EEG_noisy_EMG = np.load('EEG_noisy_EMG_bp.npy')


# %% Compute SNR for a contaminated dataset (EMG shown)
import math

def rmsValue(arr):
    """Compute RMS of a 1D array."""
    square = 0.0
    n = len(arr)
    for i in range(n):
        square += (arr[i] ** 2)
    mean = square / float(n)
    return math.sqrt(mean)

def snrValue(cleanSig, noisySig, scalingfactor):
    """Compute SNR (dB) between cleanSig and scaled noisySig."""
    return 20 * math.log10(rmsValue(cleanSig) / rmsValue(noisySig * scalingfactor))

SNRs = []
factor = 1
for i in range(len(EEG_clean_EMG)):
    SNRs.append(snrValue(EEG_clean_EMG[i, :], EEG_noisy_EMG[i, :], factor))

print("SNRs: ", SNRs)
print("SNR min: ", min(SNRs), "SNR max: ", max(SNRs))


# %% Optional: band-pass filter the clean-set (disabled by default)
# Tip: plot first to ensure signals were not previously filtered (avoid double filtering)

# fs = 200
# def bandpass_filtering(data):
#     """2nd-order Butterworth band-pass (1–50 Hz)."""
#     nyq = 0.5 * fs
#     b, a = signal.butter(2, [1/nyq, 50/nyq], btype='bandpass', analog=False)
#     return signal.filtfilt(b, a, data)

# # Example (EOG):
# for i in range(len(EEG_clean_EOG)):
#     if len(EEG_clean_EOG[i]) > 1:
#         EEG_clean_EOG[i] = bandpass_filtering(EEG_clean_EOG[i])[1*fs:-1*fs]
#         EEG_noisy_EOG[i] = EEG_noisy_EOG[i][1*fs:-1*fs]

# # Example (EMG):
# for i in range(len(EEG_clean_EMG)):
#     EEG_clean_EMG[i] = bandpass_filtering(EEG_clean_EMG[i])


# %% High-pass filter on Noisy EMG (per reviewer’s suggestion)
fs = 200
def highpass_filtering(data):
    """2nd-order Butterworth high-pass (cutoff=1 Hz)."""
    nyq = 0.5 * fs
    b, a = signal.butter(2, 1/nyq, btype='highpass', analog=False)
    return signal.filtfilt(b, a, data)

for i in range(len(EEG_noisy_EMG)):
    EEG_noisy_EMG[i] = highpass_filtering(EEG_noisy_EMG[i])
    print(i)


# %% Quick visualization (single sample from EMG)
fs = 200
time = np.linspace(0, len(EEG_clean_EMG[0]) / fs, num=len(EEG_clean_EMG[0]))

i = 5
plt.subplot(2, 1, 1)
plt.plot(time, EEG_noisy_EMG[i], label="Corrupted EEG/EMG", linewidth=2)
plt.ylabel('Corrupted')
plt.subplot(2, 1, 2)
plt.plot(time, EEG_clean_EMG[i], label="Ground truth EEG/EMG", linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Ground truth')
plt.show()


# %% Segmentation (4 s window, 50% overlap)
import math
fs = 200
def data_segment(data):
    """Segment a 1D signal into 4 s windows with 50% overlap."""
    segment_len = int(4 * fs)
    overlap = 0.5
    step = int(segment_len * (1 - overlap))
    num_segments = math.floor(len(data) / step) - 1
    segments = []
    for i in range(num_segments):
        seg = data[int(step * i): int(step * i + segment_len)]
        segments.append(seg)
    return segments

# Containers for all segmented sets
EEG_clean_EOG_segments    = []
EEG_noisy_EOG_segments    = []
EEG_clean_motion_segments = []
EEG_noisy_motion_segments = []
EEG_clean_EMG_segments    = []
EEG_noisy_EMG_segments    = []

# Segment EOG
for i in range(len(EEG_clean_EOG)):
    S_clean = data_segment(EEG_clean_EOG[i])
    S_noisy = data_segment(EEG_noisy_EOG[i])
    print('Segments (EOG item): ' + str(len(S_clean)))
    for j in range(len(S_clean)):
        EEG_clean_EOG_segments.append(S_clean[j])
        EEG_noisy_EOG_segments.append(S_noisy[j])

# Segment Motion
for i in range(len(EEG_clean_motion)):
    S_clean = data_segment(EEG_clean_motion[i])
    S_noisy = data_segment(EEG_noisy_motion[i])
    print('Segments (Motion item): ' + str(len(S_clean)))
    for j in range(len(S_clean)):
        EEG_clean_motion_segments.append(S_clean[j])
        EEG_noisy_motion_segments.append(S_noisy[j])

# Segment EMG
for i in range(len(EEG_clean_EMG)):
    S_clean = data_segment(EEG_clean_EMG[i])
    S_noisy = data_segment(EEG_noisy_EMG[i])
    print('Segments (EMG item): ' + str(len(S_clean)))
    for j in range(len(S_clean)):
        EEG_clean_EMG_segments.append(S_clean[j])
        EEG_noisy_EMG_segments.append(S_noisy[j])


# %% Per-segment min–max normalization to [0, 1]
# Keep track of min/max to enable potential de-normalization
EEG_clean_EOG_norm,   EEG_noisy_EOG_norm   = [], []
EEG_clean_motion_norm,EEG_noisy_motion_norm= [], []
EEG_clean_EMG_norm,   EEG_noisy_EMG_norm   = [], []

maxValue_clean_EOG,   minValue_clean_EOG   = [], []
maxValue_noisy_EOG,   minValue_noisy_EOG   = [], []
maxValue_clean_motion,minValue_clean_motion= [], []
maxValue_noisy_motion,minValue_noisy_motion= [], []
maxValue_clean_EMG,   minValue_clean_EMG   = [], []
maxValue_noisy_EMG,   minValue_noisy_EMG   = [], []

# EOG
for i in range(len(EEG_clean_EOG_segments)):
    data_c  = EEG_clean_EOG_segments[i]
    data_n  = EEG_noisy_EOG_segments[i]
    data_cN = np.zeros_like(data_c)
    data_nN = np.zeros_like(data_n)
    for j in range(len(data_c)):
        data_cN[j] = (data_c[j] - data_c.min()) / (data_c.max() - data_c.min())
        data_nN[j] = (data_n[j] - data_n.min()) / (data_n.max() - data_n.min())
    maxValue_clean_EOG.append(data_c.max());  minValue_clean_EOG.append(data_c.min())
    maxValue_noisy_EOG.append(data_n.max());  minValue_noisy_EOG.append(data_n.min())
    EEG_clean_EOG_norm.append(data_cN)
    EEG_noisy_EOG_norm.append(data_nN)

# Motion
for i in range(len(EEG_clean_motion_segments)):
    data_c  = EEG_clean_motion_segments[i]
    data_n  = EEG_noisy_motion_segments[i]
    data_cN = np.zeros_like(data_c)
    data_nN = np.zeros_like(data_n)
    for j in range(len(data_c)):
        data_cN[j] = (data_c[j] - data_c.min()) / (data_c.max() - data_c.min())
        data_nN[j] = (data_n[j] - data_n.min()) / (data_n.max() - data_n.min())
    maxValue_clean_motion.append(data_c.max());  minValue_clean_motion.append(data_c.min())
    maxValue_noisy_motion.append(data_n.max());  minValue_noisy_motion.append(data_n.min())
    EEG_clean_motion_norm.append(data_cN.flatten())
    EEG_noisy_motion_norm.append(data_nN.flatten())

# EMG
for i in range(len(EEG_clean_EMG_segments)):
    data_c  = EEG_clean_EMG_segments[i]
    data_n  = EEG_noisy_EMG_segments[i]
    data_cN = np.zeros_like(data_c)
    data_nN = np.zeros_like(data_n)
    for j in range(len(data_c)):
        data_cN[j] = (data_c[j] - data_c.min()) / (data_c.max() - data_c.min())
        data_nN[j] = (data_n[j] - data_n.min()) / (data_n.max() - data_n.min())
    maxValue_clean_EMG.append(data_c.max());  minValue_clean_EMG.append(data_c.min())
    maxValue_noisy_EMG.append(data_n.max());  minValue_noisy_EMG.append(data_n.min())
    EEG_clean_EMG_norm.append(data_cN)
    EEG_noisy_EMG_norm.append(data_nN)


# %% Split into Train / Val / Test (80/10/10; no shuffling pre-split)
# 1) EOG
train_clean_EOG, testEOG       = train_test_split(EEG_clean_EOG_norm,  test_size=0.2, shuffle=False)
val_clean_EOG,   test_clean_EOG= train_test_split(testEOG,             test_size=0.5, shuffle=False)
train_noisy_EOG, testEOG2      = train_test_split(EEG_noisy_EOG_norm,  test_size=0.2, shuffle=False)
val_noisy_EOG,   test_noisy_EOG= train_test_split(testEOG2,            test_size=0.5, shuffle=False)

# 2) Motion
train_clean_motion, testMotion       = train_test_split(EEG_clean_motion_norm,  test_size=0.2, shuffle=False)
val_clean_motion,   test_clean_motion= train_test_split(testMotion,             test_size=0.5, shuffle=False)
train_noisy_motion, testMotion2      = train_test_split(EEG_noisy_motion_norm,  test_size=0.2, shuffle=False)
val_noisy_motion,   test_noisy_motion= train_test_split(testMotion2,            test_size=0.5, shuffle=False)

# 3) EMG
train_clean_EMG, testEMG       = train_test_split(EEG_clean_EMG_norm,  test_size=0.2, shuffle=False)
val_clean_EMG,   test_clean_EMG= train_test_split(testEMG,             test_size=0.5, shuffle=False)
train_noisy_EMG, testEMG2      = train_test_split(EEG_noisy_EMG_norm,  test_size=0.2, shuffle=False)
val_noisy_EMG,   test_noisy_EMG= train_test_split(testEMG2,            test_size=0.5, shuffle=False)


# %% Merge modalities for unified Train/Val/Test arrays
train_clean, train_noisy = [], []
val_clean,   val_noisy   = [], []
test_clean,  test_noisy  = [], []

# Train (clean)
for x in [train_clean_EOG, train_clean_motion, train_clean_EMG]:
    for s in x: train_clean.append(s)
# Train (noisy)
for x in [train_noisy_EOG, train_noisy_motion, train_noisy_EMG]:
    for s in x: train_noisy.append(s)

# Val (clean)
for x in [val_clean_EOG, val_clean_motion, val_clean_EMG]:
    for s in x: val_clean.append(s)
# Val (noisy)
for x in [val_noisy_EOG, val_noisy_motion, val_noisy_EMG]:
    for s in x: val_noisy.append(s)

# Test (clean)
for x in [test_clean_EOG, test_clean_motion, test_clean_EMG]:
    for s in x: test_clean.append(s)
# Test (noisy)
for x in [test_noisy_EOG, test_noisy_motion, test_noisy_EMG]:
    for s in x: test_noisy.append(s)


# %% Permute paired lists (preserve alignment between clean/noisy)
import random
cc = list(zip(train_clean, val_clean))
# (Note: The original code shuffles (clean, noisy) pairs; kept as-is below.)
cc = list(zip(train_clean, train_noisy))
random.shuffle(cc)
train_clean[:], train_noisy[:] = zip(*cc)

cc = list(zip(val_clean, val_noisy))
random.shuffle(cc)
val_clean[:], val_noisy[:] = zip(*cc)


# %% Convert lists to numpy arrays
x_train_clean = np.array(train_clean)
x_train_noisy = np.array(train_noisy)
x_val_clean   = np.array(val_clean)
x_val_noisy   = np.array(val_noisy)
x_test_clean  = np.array(test_clean)
x_test_noisy  = np.array(test_noisy)


# %% Clean-segment proportion check via correlation (train/val/test)
# Idea: if corr(noisy, clean) > 0.95, the "noisy" input is likely already clean.
import math
import scipy.stats

cc_trainset = []
for i in range(len(x_train_noisy)):
    cc_trainset.append(np.corrcoef(x_train_noisy[i], x_train_clean[i])[0, 1])
cc_trainset = np.array(cc_trainset)
plt.hist(cc_trainset); plt.show()
print(sum(k > 0.95 for k in cc_trainset))

cc_testset = []
for i in range(len(x_test_noisy)):
    cc_testset.append(np.corrcoef(x_test_noisy[i], x_test_clean[i])[0, 1])
cc_testset = np.array(cc_testset)
plt.hist(cc_testset); plt.show()
print(sum(k > 0.95 for k in cc_testset))

cc_valset = []
for i in range(len(x_val_noisy)):
    cc_valset.append(np.corrcoef(x_val_noisy[i], x_val_clean[i])[0, 1])
cc_valset = np.array(cc_valset)
plt.hist(cc_valset); plt.show()
print(sum(k > 0.95 for k in cc_valset))


# %% Train a baseline 1D convolutional autoencoder
import time
start_time = time.time()

class Autoencoder(Model):
    """Simple 1D CNN autoencoder (encoder: 64->32->16->4, decoder: 16->32->64->1)."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(800, 1)),
            layers.Conv1D(64, 3, activation='relu', padding='same', strides=1),
            layers.Conv1D(32, 3, activation='relu', padding='same', strides=1),
            layers.Conv1D(16, 3, activation='relu', padding='same', strides=1),
            layers.Conv1D(4,  3, activation='relu', padding='same', strides=1)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv1D(16, 3, activation='relu', padding='same', strides=1),
            layers.Conv1D(32, 3, activation='relu', padding='same', strides=1),
            layers.Conv1D(64, 3, activation='relu', padding='same', strides=1),
            layers.Conv1D(1,  kernel_size=3, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded  = self.encoder(x)
        decoded  = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

# Early stopping (optional)
pat = 5
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

# Checkpointing (optional)
path_checkpoint    = "training_1/cp.ckpt"
directory_checkpoint = os.path.dirname(path_checkpoint)
callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                              save_weights_only=True,
                                              verbose=1)

Epochs  = 1000
history = autoencoder.fit(x_train_noisy, x_train_clean,
                          epochs=Epochs,
                          shuffle=True,
                          validation_data=(x_val_noisy, x_val_clean))

autoencoder.encoder.summary()
autoencoder.decoder.summary()

# Forward on test set
encoded_layer = autoencoder.encoder(x_test_noisy).numpy()
decoded_layer = autoencoder.decoder(encoded_layer).numpy()
decoded_layer = np.squeeze(decoded_layer)  # (N, 800, 1) -> (N, 800)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# %% Plot learning curves (Loss / Accuracy)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Learning Curve - Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Learning Curve - Accuracy'); plt.legend()
plt.tight_layout()
# plt.savefig('learning_curve.pdf')
plt.show()


# %% Save training History object (pickle)
import pickle
with open('history.pkl', 'wb') as file:
    pickle.dump(history, file)
# To reload:
# with open('history.pkl', 'rb') as file:
#     loaded_history = pickle.load(file)


# %% Export TFLite model
# Optional: save Keras SavedModel as well
# autoencoder.save('saved_model/Autoencoder_revision')
converter    = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()
with open('autoencoder_revision.tflite', 'wb') as f:
    f.write(tflite_model)


# %% Quick qualitative visualization (original vs recon)
fs   = 200
time = np.linspace(0, len(x_test_clean[0]) / fs, num=len(x_test_clean[0]))
n    = 10
plt.figure(figsize=(30, 10))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.title("original")
    plt.plot(time, x_test_noisy[i + 100, :])
    ax.get_xaxis().set_visible(True); ax.get_yaxis().set_visible(True)
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("recon")
    plt.plot(time, decoded_layer[i + 100, :])
    bx.get_xaxis().set_visible(True); bx.get_yaxis().set_visible(True)
plt.suptitle('Autoencoder Input and Output examples')
plt.show()


# %% Save arrays for later reuse (keep naming as in original script)
# x_test_noisy : denoiser inputs
# x_test_clean : ground truth clean signals
# decoded_layer: denoised outputs
np.save("x_test_noisy1",   x_test_noisy,   'datasets')
np.save("x_test_clean1",   x_test_clean,   'datasets')
np.save("decoded_layer1",  decoded_layer,  'datasets')


# %% Zero-centering before statistics (recommended if normalized to [0,1])
z_test_noisy       = np.zeros(x_test_noisy.shape)
z_test_clean       = np.zeros(x_test_clean.shape)
z_decoded_layer    = np.zeros(x_test_clean.shape)

for i in range(len(x_test_clean)):
    z_test_noisy[i]    = x_test_noisy[i] - np.mean(x_test_noisy[i])
    z_test_clean[i]    = x_test_clean[i] - np.mean(x_test_clean[i])
    z_decoded_layer[i] = decoded_layer[i].flatten() - np.mean(decoded_layer[i].flatten())


# %% Optional: amplitude scaling of reconstruction to match clean (heuristic)
for i in range(len(z_test_clean)):
    adjust_factor1 = z_test_clean[i].max() / z_decoded_layer[i].max()
    adjust_factor2 = z_test_clean[i].min() / z_decoded_layer[i].min()
    adjust_factor  = adjust_factor1
    z_decoded_layer[i] = z_decoded_layer[i] * adjust_factor


# %% Metrics: CC, RRMSE (time), RMSE, RRMSE (frequency; PSD via Welch)
import math
import scipy.stats

# 1) Pearson correlation on zero-centered signals
CC = np.zeros((len(z_test_clean), 1))
for i in range(len(z_test_clean)):
    CC[i] = np.corrcoef(z_test_clean[i], z_decoded_layer[i])[0, 1]

# Indices for modality-specific splits (per original code comments)
CC_EOG    = CC[0:345]
CC_motion = CC[345:967]
CC_EMG    = CC[967:1712]

plt.hist(CC_motion); plt.show()

# 2) RRMSE / RMSE helpers
def rmsValue(arr):
    """RMS helper (duplicated by original; kept for compatibility)."""
    square = 0.0
    n = len(arr)
    for i in range(n):
        square += (arr[i] ** 2)
    return math.sqrt(square / float(n))

def RRMSE(true, pred):
    """Relative RMSE: rms(true - pred) / rms(true)."""
    return rmsValue(true - pred) / rmsValue(true)

def RMSE(true, pred):
    """Root MSE: rms(true - pred)."""
    return rmsValue(true - pred)

# 3) RRMSE in time domain (zero-centered)
RRMSE_timeDomain = np.zeros((len(z_test_clean), 1))
for i in range(len(RRMSE_timeDomain)):
    RRMSE_timeDomain[i] = RRMSE(z_test_clean[i], z_decoded_layer[i])

RRMSE_EOG    = RRMSE_timeDomain[0:345]
RRMSE_motion = RRMSE_timeDomain[345:967]
RRMSE_EMG    = RRMSE_timeDomain[967:1712]

# 4) RRMSE in frequency domain (PSD via Welch)
nperseg = 200
nfft    = 800
PSD_len = nfft // 2 + 1

PSD_cleanEEG    = np.zeros((len(z_test_clean), PSD_len))
PSD_denoisedEEG = np.zeros((len(z_test_clean), PSD_len))

for i in range(len(z_test_clean)):
    _, pxx_c = signal.welch(z_test_clean[i],    fs=200, nperseg=nperseg, nfft=nfft)
    _, pxx_d = signal.welch(z_decoded_layer[i], fs=200, nperseg=nperseg, nfft=nfft)
    PSD_cleanEEG[i]    = pxx_c
    PSD_denoisedEEG[i] = pxx_d

RRMSE_freqDomain = np.zeros((len(z_test_clean), 1))
for i in range(len(RRMSE_freqDomain)):
    RRMSE_freqDomain[i] = RRMSE(PSD_cleanEEG[i], PSD_denoisedEEG[i])

# Note: these slice indices (372, 1003, 1748) reflect a different split note in comments.
RRMSE_EOGf    = RRMSE_freqDomain[0:372]
RRMSE_motionf = RRMSE_freqDomain[372:1003]
RRMSE_EMGf    = RRMSE_freqDomain[1003:1748]


# %% Summary (mixed clean/noisy inputs by modality)
print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ", np.mean(RRMSE_EOG),    " ,std= ", np.std(RRMSE_EOG))
print("RRMSE-Freq: mean= ", np.mean(RRMSE_EOGf),   " ,std= ", np.std(RRMSE_EOGf))
print("CC: mean= ",        np.mean(CC_EOG),        " ,std= ", np.std(CC_EOG))

print("\n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ", np.mean(RRMSE_motion),  " ,std= ", np.std(RRMSE_motion))
print("RRMSE-Freq:  mean= ", np.mean(RRMSE_motionf), " ,std= ", np.std(RRMSE_motionf))
print("CC:  mean= ",        np.mean(CC_motion),      " ,std= ", np.std(CC_motion))

print("\n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ", np.mean(RRMSE_EMG),   " ,std= ", np.std(RRMSE_EMG))
print("RRMSE-Freq:  mean= ", np.mean(RRMSE_EMGf),  " ,std= ", np.std(RRMSE_EMGf))
print("CC:  mean= ",         np.mean(CC_EMG),      " ,std= ", np.std(CC_EMG))

# Save Keras model (path kept as in original script)
model    = autoencoder
filepath = 'C:/models/Autoencoder_CNN_model_v4'
tf.keras.models.save_model(model, filepath)


# %% Reload model and cached arrays (when previously saved)
autoencoder = tf.keras.models.load_model('C:/models/Autoencoder_CNN_model_v4')
autoencoder.summary()
encoded_layer = autoencoder.encoder(x_test_noisy).numpy()
decoded_layer = autoencoder.decoder(encoded_layer).numpy()

x_test_noisy  = np.load("x_test_noisy1.npy")
x_test_clean  = np.load("x_test_clean1.npy")
decoded_layer = np.load("decoded_layer1.npy")


# %% Split test set into (detected) clean vs noisy inputs
# Criterion: CC(noisy, clean) > 0.95 -> treated as "clean input"
clean_detect, noisy_detect = [], []
CC_detectClean = np.zeros((len(z_test_clean), 1))
for i in range(len(z_test_clean)):
    CC_detectClean[i] = np.corrcoef(z_test_clean[i], z_test_noisy[i])[0, 1]
    if CC_detectClean[i] > 0.95:
        clean_detect.append(i)
    else:
        noisy_detect.append(i)

# Prepare containers
clean_inputs, clean_outputs = [], []
noisy_inputs_EOG,   noisy_outputs_EOG,   ground_truth_EOG   = [], [], []
noisy_inputs_Motion,noisy_outputs_Motion,ground_truth_Motion= [], [], []
noisy_inputs_EMG,   noisy_outputs_EMG,   ground_truth_EMG   = [], [], []

# Collect clean pairs
for idx in clean_detect:
    clean_inputs.append(z_test_noisy[idx])
    clean_outputs.append(z_decoded_layer[idx])

# Collect noisy by modality using index ranges
for idx in noisy_detect:
    if idx < 345:
        noisy_inputs_EOG.append(z_test_noisy[idx])
        noisy_outputs_EOG.append(z_decoded_layer[idx])
        ground_truth_EOG.append(z_test_clean[idx])
    elif 345 <= idx < 967:
        noisy_inputs_Motion.append(z_test_noisy[idx])
        noisy_outputs_Motion.append(z_decoded_layer[idx])
        ground_truth_Motion.append(z_test_clean[idx])
    else:
        noisy_inputs_EMG.append(z_test_noisy[idx])
        noisy_outputs_EMG.append(z_decoded_layer[idx])
        ground_truth_EMG.append(z_test_clean[idx])


# %% Save separated evaluation sets
np.save("clean_inputs1",  clean_inputs,  'datasets')
np.save("clean_outputs1", clean_outputs, 'datasets')

np.save("noisy_inputs_EOG1",   noisy_inputs_EOG,   'datasets')
np.save("noisy_outputs_EOG1",  noisy_outputs_EOG,  'datasets')
np.save("ground_truth_EOG1",   ground_truth_EOG,   'datasets')

np.save("noisy_inputs_Motion1",  noisy_inputs_Motion,  'datasets')
np.save("noisy_outputs_Motion1", noisy_outputs_Motion, 'datasets')
np.save("ground_truth_Motion1",  ground_truth_Motion,  'datasets')

np.save("noisy_inputs_EMG1",   noisy_inputs_EMG,   'datasets')
np.save("noisy_outputs_EMG1",  noisy_outputs_EMG,  'datasets')
np.save("ground_truth_EMG1",   ground_truth_EMG,   'datasets')


# %% Final evaluation (per requested paper-ready statistics)

# --- Clean reconstruction ---
clean_inputs_RRMSE, clean_inputs_RRMSEABS = [], []
for i in range(len(clean_inputs)):
    clean_inputs_RRMSE.append(RRMSE(clean_inputs[i],  clean_outputs[i]))
    clean_inputs_RRMSEABS.append(RMSE(clean_inputs[i], clean_outputs[i]))

# PSD-based (frequency-domain) errors
nperseg = 200
nfft    = 800
PSD_len = nfft // 2 + 1
clean_inputs_PSD  = np.zeros((len(clean_inputs), PSD_len))
clean_outputs_PSD = np.zeros((len(clean_inputs), PSD_len))
for i in range(len(clean_inputs)):
    _, pxx = signal.welch(clean_inputs[i],  fs=200, nperseg=nperseg, nfft=nfft);  clean_inputs_PSD[i]  = pxx
    _, pxx = signal.welch(clean_outputs[i], fs=200, nperseg=nperseg, nfft=nfft);  clean_outputs_PSD[i] = pxx

clean_inputs_PSD_RRMSE, clean_inputs_PSD_RRMSEABS = [], []
for i in range(len(clean_inputs)):
    clean_inputs_PSD_RRMSE.append(RRMSE(clean_inputs_PSD[i],  clean_outputs_PSD[i]))
    clean_inputs_PSD_RRMSEABS.append(RMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))

# Pearson CC
import scipy.stats
clean_inputs_CC = []
for i in range(len(clean_inputs)):
    result = scipy.stats.pearsonr(clean_inputs[i], clean_outputs[i])
    clean_inputs_CC.append(result.statistic)

# --- EOG denoising ---
EOG_RRMSE, EOG_RRMSEABS = [], []
for i in range(len(noisy_inputs_EOG)):
    EOG_RRMSE.append(RRMSE(ground_truth_EOG[i],  noisy_outputs_EOG[i]))
    EOG_RRMSEABS.append(RMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))

ground_truth_EOG_PSD   = np.zeros((len(noisy_inputs_EOG), PSD_len))
noisy_outputs_EOG_PSD  = np.zeros((len(noisy_inputs_EOG), PSD_len))
for i in range(len(noisy_inputs_EOG)):
    _, pxx = signal.welch(ground_truth_EOG[i], fs=200, nperseg=nperseg, nfft=nfft); ground_truth_EOG_PSD[i]  = pxx
    _, pxx = signal.welch(noisy_outputs_EOG[i], fs=200, nperseg=nperseg, nfft=nfft); noisy_outputs_EOG_PSD[i] = pxx

EOG_PSD_RRMSE, EOG_PSD_RRMSEABS, EOG_CC = [], [], []
for i in range(len(noisy_inputs_EOG)):
    EOG_PSD_RRMSE.append(RRMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
    EOG_PSD_RRMSEABS.append(RMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
    EOG_CC.append(scipy.stats.pearsonr(ground_truth_EOG[i], noisy_outputs_EOG[i]).statistic)

# --- Motion denoising ---
Motion_RRMSE, Motion_RRMSEABS = [], []
for i in range(len(noisy_inputs_Motion)):
    Motion_RRMSE.append(RRMSE(ground_truth_Motion[i],  noisy_outputs_Motion[i]))
    Motion_RRMSEABS.append(RMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))

ground_truth_Motion_PSD  = np.zeros((len(noisy_inputs_Motion), PSD_len))
noisy_outputs_Motion_PSD = np.zeros((len(noisy_inputs_Motion), PSD_len))
for i in range(len(noisy_inputs_Motion)):
    _, pxx = signal.welch(ground_truth_Motion[i], fs=200, nperseg=nperseg, nfft=nfft); ground_truth_Motion_PSD[i]  = pxx
    _, pxx = signal.welch(noisy_outputs_Motion[i], fs=200, nperseg=nperseg, nfft=nfft); noisy_outputs_Motion_PSD[i] = pxx

Motion_PSD_RRMSE, Motion_PSD_RRMSEABS, Motion_CC = [], [], []
for i in range(len(noisy_inputs_Motion)):
    Motion_PSD_RRMSE.append(RRMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
    Motion_PSD_RRMSEABS.append(RMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
    Motion_CC.append(scipy.stats.pearsonr(ground_truth_Motion[i], noisy_outputs_Motion[i]).statistic)

# --- EMG denoising ---
EMG_RRMSE, EMG_RRMSEABS = [], []
for i in range(len(noisy_inputs_EMG)):
    EMG_RRMSE.append(RRMSE(ground_truth_EMG[i],  noisy_outputs_EMG[i]))
    EMG_RRMSEABS.append(RMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))

ground_truth_EMG_PSD  = np.zeros((len(noisy_inputs_EMG), PSD_len))
noisy_outputs_EMG_PSD = np.zeros((len(noisy_inputs_EMG), PSD_len))
for i in range(len(noisy_inputs_EMG)):
    _, pxx = signal.welch(ground_truth_EMG[i], fs=200, nperseg=nperseg, nfft=nfft); ground_truth_EMG_PSD[i]  = pxx
    _, pxx = signal.welch(noisy_outputs_EMG[i], fs=200, nperseg=nperseg, nfft=nfft); noisy_outputs_EMG_PSD[i] = pxx

EMG_PSD_RRMSE, EMG_PSD_RRMSEABS, EMG_CC = [], [], []
for i in range(len(noisy_inputs_EMG)):
    EMG_PSD_RRMSE.append(RRMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))
    EMG_PSD_RRMSEABS.append(RMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))
    EMG_CC.append(scipy.stats.pearsonr(ground_truth_EMG[i], noisy_outputs_EMG[i]).statistic)

# Convert to arrays for summary/plots
clean_inputs_RRMSE        = np.array(clean_inputs_RRMSE)
clean_inputs_PSD_RRMSE    = np.array(clean_inputs_PSD_RRMSE)
clean_inputs_CC           = np.array(clean_inputs_CC)
clean_inputs_RRMSEABS     = np.array(clean_inputs_RRMSEABS)
clean_inputs_PSD_RRMSEABS = np.array(clean_inputs_PSD_RRMSEABS)

EOG_RRMSE        = np.array(EOG_RRMSE)
EOG_PSD_RRMSE    = np.array(EOG_PSD_RRMSE)
EOG_CC           = np.array(EOG_CC)
EOG_RRMSEABS     = np.array(EOG_RRMSEABS)
EOG_PSD_RRMSEABS = np.array(EOG_PSD_RRMSEABS)

Motion_RRMSE        = np.array(Motion_RRMSE)
Motion_PSD_RRMSE    = np.array(Motion_PSD_RRMSE)
Motion_CC           = np.array(Motion_CC)
Motion_RRMSEABS     = np.array(Motion_RRMSEABS)
Motion_PSD_RRMSEABS = np.array(Motion_PSD_RRMSEABS)

EMG_RRMSE        = np.array(EMG_RRMSE)
EMG_PSD_RRMSE    = np.array(EMG_PSD_RRMSE)
EMG_CC           = np.array(EMG_CC)
EMG_RRMSEABS     = np.array(EMG_RRMSEABS)
EMG_PSD_RRMSEABS = np.array(EMG_PSD_RRMSEABS)

# Print concise summaries
print("\n EEG clean input results: ")
print("RRMSE-Time: mean= ", "%.4f" % np.mean(clean_inputs_RRMSE),     " ,std= ", "%.4f" % np.std(clean_inputs_RRMSE))
print("RRMSE-Freq: mean= ", "%.4f" % np.mean(clean_inputs_PSD_RRMSE), " ,std= ", "%.4f" % np.std(clean_inputs_PSD_RRMSE))
print("CC: mean= ",        "%.4f" % np.mean(clean_inputs_CC),         " ,std= ", "%.4f" % np.std(clean_inputs_CC))

print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ", "%.4f" % np.mean(EOG_RRMSE),       " ,std= ", "%.4f" % np.std(EOG_RRMSE))
print("RRMSE-Freq: mean= ", "%.4f" % np.mean(EOG_PSD_RRMSE),   " ,std= ", "%.4f" % np.std(EOG_PSD_RRMSE))
print("CC: mean= ",         "%.4f" % np.mean(EOG_CC),          " ,std= ", "%.4f" % np.std(EOG_CC))

print(" \n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ", "%.4f" % np.mean(Motion_RRMSE),     " ,std= ", "%.4f" % np.std(Motion_RRMSE))
print("RRMSE-Freq:  mean= ", "%.4f" % np.mean(Motion_PSD_RRMSE), " ,std= ", "%.4f" % np.std(Motion_PSD_RRMSE))
print("CC:  mean= ",         "%.4f" % np.mean(Motion_CC),        " ,std= ", "%.4f" % np.std(Motion_CC))

print(" \n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ", "%.4f" % np.mean(EMG_RRMSE),      " ,std= ", "%.4f" % np.std(EMG_RRMSE))
print("RRMSE-Freq:  mean= ", "%.4f" % np.mean(EMG_PSD_RRMSE),  " ,std= ", "%.4f" % np.std(EMG_PSD_RRMSE))
print("CC:  mean= ",         "%.4f" % np.mean(EMG_CC),         " ,std= ", "%.4f" % np.std(EMG_CC))


# %% Boxplots (optional visual summaries)
plt.boxplot([clean_inputs_RRMSE, clean_inputs_PSD_RRMSE, clean_inputs_CC])
plt.title("Clean EEG input"); plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Spectral','CC'])

plt.boxplot([EOG_RRMSE, EOG_PSD_RRMSE, EOG_CC])
plt.title("EEG/EOG artifacts"); plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Spectral','CC'])

plt.boxplot([Motion_RRMSE, Motion_PSD_RRMSE, Motion_CC])
plt.title("EEG/Motion artifacts"); plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Spectral','CC'])

plt.boxplot([EMG_RRMSE, EMG_PSD_RRMSE, EMG_CC])
plt.title("EEG/EMG artifacts"); plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Spectral','CC'])


# %% Export single examples used in papers (clean)
fs   = 200
time = np.linspace(0, len(x_test_clean[0]) / fs, num=len(x_test_clean[0]))

i = 25  # examples: 2, 5, 8, 25
plt.plot(time, clean_inputs[i],  label="Clean EEG input",           linewidth=2)
plt.plot(time, clean_outputs[i], label="Clean EEG reconstruction",  color='orange', linestyle='dashed', linewidth=1)
plt.legend(); plt.xlabel('Time (s)'); plt.ylabel(r'Normalized amplitude ($\mu$V)')
plt.tight_layout(); plt.savefig('cleanEEG_single4.pdf'); plt.show()


# %% Export single examples (artifacts)
# EOG candidates: 4, 45, 32, 27, 258
# Motion candidates: 109, 156, 16, 110–119
# EMG candidates: 1, 6, 21, 29
i = 110
plt.plot(time, ground_truth_Motion[i], label="Ground-truth clean EEG")
plt.plot(time, noisy_inputs_Motion[i], label="Contaminated EEG")
plt.plot(time, noisy_outputs_Motion[i], label="DAE denoised EEG", linestyle='dashed', linewidth=1.5)
plt.legend(); plt.xlabel('Time (s)'); plt.ylabel(r'Normalized amplitude ($\mu$V)')
plt.tight_layout(); plt.savefig('Motion_single4.pdf'); plt.show()


# %% Grid of clean reconstructions
fs   = 200
time = np.linspace(0, len(x_test_clean[0]) / fs, num=len(x_test_clean[0]))
n = 8
plt.figure(figsize=(20, 10))
for i in range(n):
    plt.subplot(2, 4, i + 1)
    plt.plot(time, clean_inputs[i + 40],  label="Clean EEG input",          linewidth=2)
    plt.plot(time, clean_outputs[i + 40], label="Clean EEG reconstruction", color='orange', linestyle='dashed', linewidth=1)
    plt.xlabel('Time (s)'); plt.ylabel(r'Normalized amplitude'); plt.rcParams.update({'font.size': 15})
    if i == 0: plt.legend()
plt.tight_layout(); plt.savefig('cleanEEG2.pdf'); plt.show()
# Note: savefig should precede show() to avoid blank outputs.


# %% Grid of EMG denoising examples
fs   = 200
time = np.linspace(0, len(x_test_clean[0]) / fs, num=len(x_test_clean[0]))
plt.figure(figsize=(20, 10))
n = 8
for i in range(n):
    plt.subplot(2, 4, i + 1)
    plt.plot(time, ground_truth_EMG[i + 33], label="Ground-truth clean EEG")
    plt.plot(time, noisy_inputs_EMG[i + 33], label="Contaminated EEG")
    plt.plot(time, noisy_outputs_EMG[i + 33], label="DAE denoised EEG", linestyle='dashed', linewidth=1.5)
    plt.legend(); plt.xlabel('Time (s)'); plt.ylabel(r'Normalized amplitude ($\mu$V)')
plt.tight_layout()
# plt.savefig('EMGplots88.pdf')
plt.show()


# %% EMG figure with stacked panels (input vs recon vs GT)
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(20, 10))
gs0 = gridspec.GridSpec(2, 4, figure=fig)
idx = 48  # EMG starting index
for i in range(8):
    gs00 = gs0[i].subgridspec(3, 1)
    ax1 = fig.add_subplot(gs00[0:1, :])
    plt.plot(time, noisy_inputs_EMG[i + idx] * 800, label="Contaminated EEG", color='tab:blue')
    ax1.get_xaxis().set_visible(False)
    plt.ylabel(r'Amplitude ($\mu$V)'); plt.rcParams.update({'font.size': 16})
    if i == 0: plt.legend()

    ax2 = fig.add_subplot(gs00[1:3, :])
    plt.plot(time, ground_truth_EMG[i + idx],  label="Ground-truth clean EEG", color='tab:green')
    plt.plot(time, noisy_outputs_EMG[i + idx], label="Reconstructed EEG", linestyle='dashed', linewidth=1.5, color='tab:orange')
    plt.xlabel('Time (s)'); plt.ylabel(r'Normalized amplitude'); plt.rcParams.update({'font.size': 16})
    if i == 0: plt.legend()
plt.tight_layout(); plt.savefig('EMGplots4.pdf'); plt.show()


# %% Triple-row visualization: input vs GT vs recon (qualitative)
fs   = 200
time = np.linspace(0, len(x_test_clean[0]) / fs, num=len(x_test_clean[0]))
n    = 10
plt.figure(figsize=(30, 10))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.title("original");     plt.plot(time, x_test_noisy[i + 400, :])
    ax.get_xaxis().set_visible(True); ax.get_yaxis().set_visible(True)
    bx = plt.subplot(3, n, i + n + 1)
    plt.title("ground-truth"); plt.plot(time, x_test_clean[i + 400, :])
    bx.get_xaxis().set_visible(True); bx.get_yaxis().set_visible(True)
    cx = plt.subplot(3, n, i + n * 2 + 1)
    plt.title("recon");        plt.plot(time, decoded_layer[i + 400, :])
    cx.get_xaxis().set_visible(True); cx.get_yaxis().set_visible(True)
plt.suptitle('Autoencoder Input and Output examples'); plt.show()

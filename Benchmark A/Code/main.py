import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
from data_prepare import *
from Network_structure import *
from loss_function import *
from train_method import *
from save_method import *
import sys
import os


# ==============================================================
# EEGdenoiseNet V2 - Main Training Script
# Author: Haoming Zhang
# Modified by: Shi Cheng
#
# This script defines the main pipeline for training denoising
# neural networks on EEG data contaminated with artifacts.
# Users can adjust parameters in the user-defined section.
# ==============================================================


###############################################################
# User-defined parameters
###############################################################

epochs = 50              # Number of training epochs
batch_size = 40          # Batch size for training
combin_num = 10          # Number of combinations of EEG and noise
denoise_network = 'fcNN' # Options: 'fcNN', 'Simple_CNN', 'Complex_CNN', 'RNN_lstm', 'Novel_CNN'
noise_type = 'EMG'       # Artifact type: 'EOG' or 'EMG'

# Directory for saving results
result_location = r'C:/models/FCNN'   # !!! Change to your own save path
foldername = 'EMG_unet112dense_10_rmsp_test'  # Subfolder name for this experiment run

# GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Whether to save signals for train/validation/test sets
save_train = True
save_vali = True
save_test = True


###############################################################
# Optimizer configuration
###############################################################

rmsp = tf.optimizers.RMSprop(lr=0.00005, rho=0.9)
adam = tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd = tf.optimizers.SGD(lr=0.0002, momentum=0.9, decay=0.0, nesterov=False)

# Select optimizer
optimizer = rmsp

# Input dimension (number of EEG samples per segment)
if noise_type == 'EOG':
    datanum = 512
elif noise_type == 'EMG':
    datanum = 512


###############################################################
# Example: Load a previously trained network (optional)
###############################################################
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''


###############################################################
# Data import
###############################################################

file_location = 'C:/Users/35152/Desktop/EEEN60070/EEG (After 6.17)/Review Work/EEGdenoiseNet-master/EEGdenoiseNet-master/data/'
# !!! Change to your own data path

if noise_type == 'EOG':
    EEG_all = np.load(file_location + 'EEG_all_epochs.npy')
    noise_all = np.load(file_location + 'EOG_all_epochs.npy')
elif noise_type == 'EMG':
    EEG_all = np.load(file_location + 'EEG_all_epochs.npy')
    noise_all = np.load(file_location + 'EMG_all_epochs.npy')


###############################################################
# Data preparation
###############################################################

# Run training multiple times to improve statistical power
i = 1

noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(
    EEG_all=EEG_all,
    noise_all=noise_all,
    combin_num=10,
    train_per=0.8,
    noise_type=noise_type
)

# Alternative: Load preprocessed data directly
# data_folder = r"C:/models/Processed_data/EMG"
# noiseEEG_train = np.load(os.path.join(data_folder, 'noiseinput_train.npy'))
# EEG_train       = np.load(os.path.join(data_folder, 'EEG_train.npy'))
# noiseEEG_val    = np.load(os.path.join(data_folder, 'noiseinput_val.npy'))
# EEG_val         = np.load(os.path.join(data_folder, 'EEG_val.npy'))
# noiseEEG_test   = np.load(os.path.join(data_folder, 'noiseinput_test.npy'))
# EEG_test        = np.load(os.path.join(data_folder, 'EEG_test.npy'))


###############################################################
# Network selection
###############################################################

if denoise_network == 'fcNN':
    model = fcNN(datanum)
elif denoise_network == 'Simple_CNN':
    model = simple_CNN(datanum)
elif denoise_network == 'Complex_CNN':
    model = Complex_CNN(datanum)
elif denoise_network == 'RNN_lstm':
    model = RNN_lstm(datanum)
else:
    print('Error: Unknown network name')


###############################################################
# Training
###############################################################

saved_model, history = train(
    model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
    epochs, batch_size, optimizer, denoise_network,
    result_location, foldername, train_num=str(i)
)


###############################################################
# Testing
###############################################################

denoised_test, test_mse = test_step(saved_model, noiseEEG_test, EEG_test)


###############################################################
# Save signals and results
###############################################################

save_eeg(
    saved_model, result_location, foldername,
    save_train, save_vali, save_test,
    noiseEEG_train, EEG_train,
    noiseEEG_val, EEG_val,
    noiseEEG_test, EEG_test,
    train_num=str(i)
)

np.save(result_location + '/' + foldername + '/' + str(i) + '/' + "nn_output" + '/loss_history.npy', history)


###############################################################
# Save trained model (optional)
###############################################################
# path = os.path.join(result_location, foldername, str(i+1), "denoise_model")
# tf.keras.models.save_model(saved_model, path)

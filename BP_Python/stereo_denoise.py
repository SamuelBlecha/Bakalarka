# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:04:52 2023

@author: Admin
"""

import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import torch.nn.functional as F
import matplotlib.pyplot as plt

file_path = './Audio/Crab_Rave.wav'
output_file_path = './Audio/new_file.wav'

# Load the audio file
sample_rate, waveform = wavfile.read(file_path)
audio_tensor = torch.FloatTensor(waveform.T).view(2, -1)

"""
# Check if the length is a multiple of sample width and channels
extra_samples = len(numpy_waveform) % (numpy_waveform.dtype.itemsize * tensor.shape[1])
print("numpy_waveform.dtype.itemsize:", numpy_waveform.dtype.itemsize)
print("num of samples", tensor.shape[1])
print("extra_samples:", extra_samples)

# If not a multiple, trim the waveform to make it compatible
if extra_samples > 0:
    numpy_waveform = numpy_waveform[:-extra_samples]
"""

def tensor_to_wav_stereo(tensor, path, sple_rate):
    original_waveform = tensor.t()
    numpy_waveform = original_waveform.detach().numpy().astype(np.int32)
    wavfile.write(output_file_path, sample_rate, numpy_waveform)
    
    
#tensor_to_wav_stereo(audio_tensor, output_file_path, sample_rate)

    
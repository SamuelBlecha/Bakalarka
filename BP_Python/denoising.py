import wave
import numpy as np
import pandas as pd
#import os

def wav_to_array(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get audio parameters
        #bit_depth = wav_file.getsampwidth()
        #sample_rate = wav_file.getframerate()
        number_of_samples = wav_file.getnframes()
        channels = wav_file.getnchannels()

        # Read audio data
        audio_data = wav_file.readframes(number_of_samples)

    # Convert binary data to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    return audio_array

# Example usage

file_path = '..\Audio\Crab_Rave.wav'
audio_array = wav_to_array(file_path)

df_array=pd.DataFrame(audio_array)
df_array.to_csv('data_array.csv')

# Print some information




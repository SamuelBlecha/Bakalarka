# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:40:30 2023

@author: Admin
"""
import numpy as np
import torch
import torchaudio
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

file_path = './Audio/Crab_rave_mono.wav'

def plot_waveform(waveform, sample_rate):
    waveform = waveform.detach().numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")


# Nacitanie audio suboru
waveform, sample_rate = torchaudio.load(file_path)

# Konverzia na 1d tenzor kde riadky predstavuju jednotlive sample
audio_tensor = torch.FloatTensor(waveform).view(1, -1)

"""
print('noise waveform:')
plot_waveform(audio_tensor, sample_rate)
torchaudio.save('audio.wav', audio_tensor, sample_rate)
"""

# Check the shape of the reshaped waveform tensor
print("Reshaped waveform shape:", audio_tensor.shape)

#vytvorenie sumu ako vstup pre neuronovu siet
num_rows, num_cols = audio_tensor.shape
noise_tensor = 2 * torch.rand(num_rows, num_cols) - 1 #nahodne cisla od 0 do 1 nasledne skalovane na -1 az +1

"""
plot_waveform(noise_tensor, sample_rate)
torchaudio.save('noise.wav', noise_tensor, sample_rate)
print('noise waveform:')
"""

# scitanie sumu a audia
noisy_audio_tensor = (noise_tensor*0.05 + audio_tensor)

"""
plot_waveform(noisy_audio_tensor, sample_rate)
torchaudio.save('noise_plus_audio.wav', noisy_audio_tensor, sample_rate)
print('noisy audio waveform:')
"""

class AudioDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.input_tensor)

# Vytvorenie datasetu and dataloaderu
dataset = AudioDataset(noise_tensor, audio_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)



class OneD_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(OneD_CNN, self).__init__()

        # Prva konvolucna vrstva
        self.conv1 = nn.Conv1d(in_channels, 4, kernel_size, stride=stride, padding=padding)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)

        # Podvzorkovanie
        self.downsample1 = nn.Conv1d(4, 4, kernel_size=2, stride=2)

        # Druha konvolucna vrstva        
        self.conv2 = nn.Conv1d(4, 8, kernel_size, stride=stride, padding=padding)
        self.BatchNorm1 = nn.BatchNorm1d(84907)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)

        # Podvzorkovanie
        self.downsample2 = nn.Conv1d(8, 8, kernel_size=2, stride=2)

        # Tretia konvolucna vrstva
        self.conv3 = nn.Conv1d(8, 16, kernel_size, stride=stride, padding=padding)
        self.BatchNorm2 = nn.BatchNorm1d(42453)
        self.leaky_relu3 = nn.LeakyReLU(inplace=True)

        # Nadvzorkovanie
        self.upsample1 = nn.ConvTranspose1d(16, 16, kernel_size=2, stride=2, output_padding=1)

        # Stvrta konvolucna vrstva
        self.conv4 = nn.Conv1d(16, 4, kernel_size, stride=stride, padding=padding)
        self.BatchNorm3 = nn.BatchNorm1d(84907)
        self.leaky_relu4 = nn.LeakyReLU(inplace=True)

        # Nadvzorkovanie
        self.upsample2 = nn.ConvTranspose1d(4, 4, kernel_size=2, stride=2)

        # posledna konvolucna vrstva
        self.conv_final = nn.Conv1d(4, in_channels, kernel_size, stride=stride, padding=padding)        
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):        
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        x = self.downsample1(x)

        x = self.BatchNorm1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)

        x = self.downsample2(x)

        x = self.BatchNorm2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)

        x = self.upsample1(x)

        x = self.conv4(x)
        x = self.BatchNorm3(x)
        x = self.leaky_relu4(x)

        x = self.upsample2(x)

        x = self.conv_final(x)
        #x = self.sigmoid(x)

        return x

"""
num_of_rows, num_of_columns = noise_tensor.shape
input_size = num_of_columns
output_size = num_of_columns  
""" 

in_channels = 1
out_channels = 1
kernel_size = 3
stride = 1
padding = 1


model = OneD_CNN(in_channels, out_channels, kernel_size, stride, padding)

#training

learning_rate = 0.0001
num_epochs = 3000
export_every = 100

loss_over_time = np.zeros(num_epochs)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    #export medzistupnov
    if epoch % export_every == 0:
        intermediate_output = model(noise_tensor)
        out_file_path = f'./Audio/iteration_{epoch}.wav'
        torchaudio.save(out_file_path, intermediate_output.detach(), sample_rate)
        print(f'exported file iteration_: {epoch}')
    for batch_input, batch_target in dataloader:
        optimizer.zero_grad()
        output = model(batch_input)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        
    loss_over_time[epoch] = loss
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}')

# Export vysledneho audio suboru
final_output = model(noise_tensor)
torchaudio.save('./Audio/final_output.wav', final_output.detach(), sample_rate)

#Graf zobrazujuci learning rate siete
plt.figure(figsize=(15, 15))
plt.plot(loss_over_time)
plt.title('Learning rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


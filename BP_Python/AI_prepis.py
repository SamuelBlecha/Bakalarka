import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import matplotlib.pyplot as plt

file_path = './crab_rave_mono.wav'
output_file_path = './new_file.wav'

# Load the audio file using librosa
sample_rate, waveform = wavfile.read('./crab_rave_mono.wav')

# Convert the waveform to a PyTorch tensor
audio_tensor = torch.FloatTensor(waveform).view(1, -1)

# Check the shape of the reshaped waveform tensor
print("Reshaped waveform shape:", audio_tensor.shape)

# Create noise tensor as input for the neural network
num_rows, num_cols = audio_tensor.shape
noise_tensor = torch.randn(1, num_cols)

# Combine noise and audio tensors
noisy_audio_tensor = noise_tensor + audio_tensor


def tensor_to_wav(tensor, path):
    original_waveform = tensor.t()
    numpy_waveform = original_waveform.detach().numpy()
    extra_samples = len(numpy_waveform) % (numpy_waveform.dtype.itemsize * 1)
    if extra_samples > 0:
        numpy_waveform = numpy_waveform[:-extra_samples]
    wavfile.write(path, sample_rate, numpy_waveform.astype(np.int32))


def plot_tensor_as_waveform(tensor):
    original_waveform = tensor.t()
    numpy_waveform = original_waveform.numpy()
    plt.figure(figsize=(15, 5))
    plt.plot(numpy_waveform[0])
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.show()


class OneD_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(OneD_CNN, self).__init__()

        # Initial 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)

        # Downsample by 2
        self.downsample1 = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)

        # Second 1D convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)

        # Downsample by 2
        self.downsample2 = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)

        # Third 1D convolutional layer
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.leaky_relu3 = nn.LeakyReLU(inplace=True)

        # Upsample by 2
        self.upsample1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2)

        # Fourth 1D convolutional layer
        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.leaky_relu4 = nn.LeakyReLU(inplace=True)

        # Upsample by 2
        self.upsample2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2)

        # Final 1D convolutional layer
        self.conv_final = nn.Conv1d(out_channels, in_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        # Downsample
        x = self.downsample1(x)

        # Second block
        x = self.conv2(x)
        x = self.leaky_relu2(x)

        # Downsample
        x = self.downsample2(x)

        # Third block
        x = self.conv3(x)
        x = self.leaky_relu3(x)

        # Upsample
        x = self.upsample1(x)

        # Fourth block
        x = self.conv4(x)
        x = self.leaky_relu4(x)

        # Upsample
        x = self.upsample2(x)

        # Final block
        x = self.conv_final(x)

        return x


# Assuming 1 channel for a grayscale signal
in_channels = 1
out_channels = 16  # Adjust as needed
kernel_size = 3
stride = 1
padding = 1

model = OneD_CNN(in_channels, out_channels, kernel_size, stride, padding)

# Training
learning_rate = 0.001
num_of_iterations = 500

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_of_iterations):
    next_step = model(noise_tensor)

    # Loss calculation
    loss = criterion(next_step, audio_tensor)

    optimizer.zero_grad()
    # Gradients = backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    print("Output tensor shape:", next_step.shape)

    if epoch % 100 == 0:
        out_file_path = f'iteration_{epoch}.wav'
        tensor_to_wav(next_step, out_file_path)

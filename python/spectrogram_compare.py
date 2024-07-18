import numpy as np
import torch
import torchaudio
import functorch as TF
import matplotlib.pyplot as plt
import cv2
from scipy.fft import fft
import pandas as pd

N_FFT = 2048
HOP_LEN = 128
SAMPLE_RATE = 48000
N_MELS = 64


def stft_to_spec(_stft):
    return _stft.abs().pow(2)


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def spec_to_mel(spec):
    spec = spec[0]
    return torch.Tensor((spec.numpy().T @ mel_fb()).T)


def mel_fb():
    n_stft = N_FFT // 2 + 1
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, n_stft).astype(np.float32)

    min_mels = hz_to_mel(0)
    max_mels = hz_to_mel(SAMPLE_RATE // 2)

    m_pts = np.linspace(min_mels, max_mels, N_MELS + 2).astype(np.float32)
    f_pts = mel_to_hz(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
    # create overlapping triangles
    zero = np.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = np.maximum(zero, np.minimum(down_slopes, up_slopes))
    return fb


def hanning_window(N):
    """
    Custom implementation of the Hanning window function.
    """
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)


def stft_compute(signal, window_size, hop_size):
    signal = signal.numpy()[0]
    signal = np.pad(signal, (window_size//2, window_size//2), 'reflect')

    """
    Compute the Short-Time Fourier Transform (STFT).
    """
    # Apply the Hanning window
    window = hanning_window(window_size)

    # Compute the number of chunks
    n_chunks = int(len(signal) / hop_size) - 1

    # Initialize the STFT matrix
    stft_matrix = []

    for i in range(n_chunks):
        # Extract the chunk
        if i >= 22471:
            a=1
        chunk = signal[i * hop_size: i * hop_size + window_size]

        if len(chunk) != window_size:
            continue
        # Apply the window
        windowed_chunk = chunk * window

        chunk_transform = np.fft.fft(windowed_chunk, n=window_size)

        # Append the chunk transform to the STFT matrix
        stft_matrix.append(chunk_transform)

    tensor = torch.from_numpy(np.array(stft_matrix)[np.newaxis, :, :])[:, :, :window_size // 2+1].transpose(1, 2).type(torch.complex64)
    return tensor


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LEN,
    n_mels=N_MELS
)

spectrogram = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP_LEN
)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
arr = np.pad(arr, (3, 3), 'reflect')

mel_scale = torchaudio.transforms.MelScale(
    64, SAMPLE_RATE, n_stft=N_FFT // 2 + 1)

wave, sr = torchaudio.load("../datasets/breathingSet2/audio/audioSample1717800518987.wav")

csh = pd.read_csv("../mel.csv", header=None)
csh = csh.to_numpy()

rawFFT = np.fft.fft(wave[0, :4096].numpy(), n=2048)
otherRaw = np.fft.fft(wave[0, :2048].numpy())



resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
wave = resampler(wave)
mel = mel_spectrogram(wave[:,:30000])
spec = spectrogram(wave)
stft = torch.stft(wave, n_fft=N_FFT, hop_length=HOP_LEN, win_length=N_FFT, window=torch.hann_window(N_FFT),
                  return_complex=True)

my_stft = stft_compute(wave, N_FFT, HOP_LEN)

mel_sp = spec_to_mel(stft_to_spec(stft))
my_mel = spec_to_mel(stft_to_spec(my_stft))

cv2.imshow("torch", cv2.resize(mel_sp.numpy(), (500, 400)))
cv2.imshow("mine", cv2.resize(my_mel.numpy(), (500, 400)))

plt.imshow(mel_sp.numpy())
plt.show()
plt.imshow(my_mel.numpy())
plt.show()
cv2.waitKey(0)
print("done")

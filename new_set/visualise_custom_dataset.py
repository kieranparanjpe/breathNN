import math
import numpy as np
import torchaudio
import csv
import os
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch
import torch.nn.functional as tF
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
CLIP_LENGTH = 60
NUM_SAMPLES = TARGET_SAMPLE_RATE * CLIP_LENGTH
AUDIO_DIRECTORY = ".\\breathingSet2\\audio"


def _resample_if_necessary(_signal, sr):
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE).to(device)
        _signal = resampler(_signal).to(device)
    return _signal


def _refine_range(_signal, index):
    index = math.floor(index % ((TARGET_SAMPLE_RATE / NUM_SAMPLES) * CLIP_LENGTH))

    _signal = _signal[:, index * NUM_SAMPLES:(index + 1) * NUM_SAMPLES]
    return _signal


def _mix_down_if_necessary(_signal):
    if _signal.shape[0] > 1:
        _signal = torch.mean(_signal, dim=0, keepdim=True)
    return _signal


def _get_paths(index):
    files = os.listdir(AUDIO_DIRECTORY)
    label_file = files[index * 2]
    audio_file = files[index * 2 + 1]  # +1 because .wav comes after .csv

    return os.path.join(AUDIO_DIRECTORY, audio_file), os.path.join(AUDIO_DIRECTORY, label_file)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    ).to(device)

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        hop_length=512
    ).to(device)

    INDEX = 3

    audio_name, audio_label = _get_paths(INDEX)
    _signal, sr = torchaudio.load(audio_name)

    _signal = _signal.to(device)
    _signal = _resample_if_necessary(_signal, sr)
    _signal = _refine_range(_signal, INDEX)
    _signal = _mix_down_if_necessary(_signal)
    shape = _signal.shape

    _raw = _signal[0].cpu().clone().numpy()

    rms = np.sqrt(np.mean(_raw ** 2))
    last = False
    for i, s in enumerate(_raw):
        if last != abs(s) > rms:
            _raw[i] = 10

        last = abs(s) > rms

    # plt.plot(_raw)
    # plt.show()

    _mel_spectrogram_signal = mel_spectrogram(_signal)
    _spectrogram_signal = spectrogram(_signal)

    _spectrogram_signal[0,0:55, :] = 0

    _mel_spectrogram_signal = torchaudio.transforms.AmplitudeToDB()(_mel_spectrogram_signal)
    _spectrogram_signal = torchaudio.transforms.AmplitudeToDB()(_spectrogram_signal)

    width = _mel_spectrogram_signal.shape[2]

    # csv labels
    annotations = pd.read_csv(audio_label, header=None)
    for row in annotations.iterrows():
        for index, point in enumerate(row[1]):
            x_coord = math.floor((int(point) / (CLIP_LENGTH * 1000)) * width)
            _mel_spectrogram_signal[0, :, x_coord - 1:x_coord + 1] = 60 * (index % 2)

    # auto labels
    square = _spectrogram_signal[0] ** 2
    rmsCol = (square.mean(axis=0)).sqrt()
    rmsRow = (square.mean(axis=1)).sqrt()

    mean = _spectrogram_signal[0].mean(axis=0)

    rmsAll = rmsCol.mean()

    # plt.plot(mean.cpu())
    # plt.show()
    rmsCol.to("cpu")

    smooth_window = 15
    smooth_rms = np.zeros(math.floor(rmsCol.shape[0] / smooth_window))
    draw_labels = np.zeros(math.floor(rmsCol.shape[0] / smooth_window))

    for i in range(smooth_rms.shape[0]):
        smooth_rms[i] = float(mean[i * smooth_window: i * smooth_window + smooth_window].mean().cpu())

    lastMean = -10000
    tot = mean.mean().cpu()
    for i in range(smooth_rms.shape[0]):
        if lastMean < tot and smooth_rms[i] >= tot or lastMean >= tot and smooth_rms[i] < tot:
            draw_labels[i] = 65
            _spectrogram_signal[0, :, i * smooth_window] = 60
            print(i)

        lastMean = smooth_rms[i]

    plt.plot(draw_labels)
    plt.plot(smooth_rms)
    plt.show()

    '''lastOver = False
    
    for index, rms in enumerate(rmsCol):
        over = rms > rmsAll
        if over != lastOver:
            _spectrogram_signal[0, :, index] = 0

        lastOver = over'''

    plt.figure(figsize=(10, 6))
    plt.imshow(_mel_spectrogram_signal.cpu()[0], aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(_spectrogram_signal.cpu()[0], aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

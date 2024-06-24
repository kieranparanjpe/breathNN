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
import cv2

TARGET_SAMPLE_RATE = 16000
CLIP_LENGTH = 60
NUM_SAMPLES = TARGET_SAMPLE_RATE * CLIP_LENGTH
AUDIO_DIRECTORY = "..\\datasets\\breathingSet2\\audioMyAnnotations"


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

    for i in range(0, 19):
        INDEX = i

        audio_name, audio_label = _get_paths(INDEX)
        _signal, sr = torchaudio.load(audio_name)

        _signal = _signal.to(device)
        _signal = _resample_if_necessary(_signal, sr)
        _signal = _refine_range(_signal, INDEX)
        _signal = _mix_down_if_necessary(_signal)
        shape = _signal.shape

        _mel_spectrogram_signal = mel_spectrogram(_signal)
        _spectrogram_signal = spectrogram(_signal)

        _spectrogram_signal[0, 0:55, :] = 0

        _mel_spectrogram_signal = torchaudio.transforms.AmplitudeToDB()(_mel_spectrogram_signal)
        _spectrogram_signal = torchaudio.transforms.AmplitudeToDB()(_spectrogram_signal)

        width = _mel_spectrogram_signal.shape[2]

        _spectrogram_with_labels = _spectrogram_signal.clone()
        # csv labels
        annotations = pd.read_csv(audio_label, header=None)
        for row in annotations.iterrows():
            for index, point in enumerate(row[1]):
                x_coord = math.floor((int(point) / (CLIP_LENGTH * 1000)) * width)
                _spectrogram_with_labels[0, :, x_coord - 1:x_coord + 1] = 60 * (index % 2)

        show_labels = False
        while True:
            img = _spectrogram_signal.cpu()[0].numpy() if not show_labels else _spectrogram_with_labels.cpu()[0].numpy()

            np_img = np.flipud(img)
            np_img = cv2.resize(np_img, (1280, 720))

            old_min, old_max = np.min(np_img), np.max(np_img)
            new_min, new_max = 0, 1

            np_img = ((np_img - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

            cv2.imshow("window", np_img)

            key = cv2.waitKey(0)
            if key == ord("q"):
                exit()
            elif key == ord("e"):
                break
            elif key == ord("z"):
                show_labels = not show_labels

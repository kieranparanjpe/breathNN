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
from trainA import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES


class BreathSet1(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device,
                 clip_length, remove_from_set):
        self.annotations = pd.read_csv(annotations_file, header=None)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.clipLength = clip_length
        self.target_dimension_2 = 0
        self.remove_from_set = remove_from_set
        self.__getitem__(0)

    def __len__(self):
        return round((len(self.annotations) - self.remove_from_set) * self.clipLength / (
                    self.num_samples / self.target_sample_rate))

    def __getitem__(self, index):
        file_index = math.floor((self.num_samples / (self.target_sample_rate * self.clipLength)) * index)

        audio_sample_path = self._get_audio_sample_path(file_index)

        sample_timestamp = ((self.num_samples / self.target_sample_rate) * index) % (self.clipLength)

        label = self._get_audio_sample_label(file_index, sample_timestamp)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._refine_range(signal, index)
        signal = self._mix_down_if_necessary(signal)
        shape = signal.shape
        signal = self.transformation(signal)
        if index == 0:
            self.target_dimension_2 = signal.shape[2]
        if signal.shape[2] != self.target_dimension_2:
            signal = tF.pad(signal, (0, self.target_dimension_2 - signal.shape[2]))
            #print(f"index: {index}, old shape: {shape}, new shape: {signal.shape}") #debug code
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal).to(self.device)
        return signal

    def _refine_range(self, signal, index):
        index = math.floor(index % ((self.target_sample_rate / self.num_samples) * self.clipLength))

        signal = signal[:, index * self.num_samples:(index + 1) * self.num_samples]
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, file_index, timestamp):
        row = self.annotations.iloc[file_index]
        for i in range(1, len(row) - 1, 2):
            if row[i] == 0:
                break
            lo, hi = float(row[i]), float(row[i + 1])
            mid = (lo + hi) / 2
            third = (hi - lo) / 3
            #print(mid, third, timestamp)
            if mid - third < timestamp <= mid:
                return 0
            if mid < timestamp < mid + third:
                return 1

        return 2


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    )

    bs1 = BreathSet1(ANNOTATIONS_FILE,
                     AUDIO_DIR,
                     mel_spectrogram,
                     SAMPLE_RATE,
                     NUM_SAMPLES,
                     device,
                     20, 0)

    print(f"There are {len(bs1)} samples in the dataset.")

    signal, label = bs1[950]
    print(label)
    ls = [0, 0, 0]
    # some of them have 2 in their -> based on clip length, some of them are at the end of the clip so its not the full length. can just pad or use samplerate divisible by num samples
    for i in tqdm(range(0, 1)):
        s, l = bs1[i]
        ls[l] += 1

    print(ls)

    print(signal.shape)
    plt.figure(figsize=(10, 6))
    plt.imshow(signal.cpu()[0].log2(), aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

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


class BreathSet2(Dataset):

    def __init__(self, audio_dir, target_sample_rate, num_samples, sample_step,
                 transformation=None, _device="cpu", clip_length=60, remove_from_set=0, cutoff_point=0,
                 load_in_ram=False, max_silence=100000000):
        self.samples = []

        self.audio_dir = audio_dir
        self.device = _device
        self.transformation = transformation.to(self.device) if transformation is not None else None

        self.target_sample_rate = target_sample_rate
        self.sample_step = sample_step
        self.num_samples = num_samples

        self.clipLength = clip_length
        self.target_dimension_2 = 0
        self.remove_from_set = remove_from_set

        self.cutoff_point = cutoff_point

        self.files = os.listdir(self.audio_dir)
        self.num_audio_files = len(
            [item for item in self.files if os.path.isfile(os.path.join(self.audio_dir, item))]) / 2

        self.load_in_ram = load_in_ram

        self.items_per_clip = (self.target_sample_rate * self.clipLength - self.num_samples) / self.sample_step

        if self.load_in_ram:
            self.audio_files_in_ram = {}
            self.csv_files_in_ram = {}

            for i in range(int(self.num_audio_files)):
                audio_file, csv_file = self._get_paths(i)
                _signal, sr = torchaudio.load(audio_file)
                _signal = _signal.to(self.device)
                _signal = self._resample_if_necessary(_signal, sr)
                _signal = self._mix_down_if_necessary(_signal)
                _signal = tF.pad(_signal, (0, self.target_sample_rate * self.clipLength - _signal.shape[1]))
                self.audio_files_in_ram[audio_file] = _signal

                annotations = pd.read_csv(csv_file, header=None)
                annotations_arr = annotations.to_numpy().flatten()
                if annotations_arr[len(annotations_arr) - 1] == 0:
                    annotations_arr = annotations_arr[0:-4]

                self.csv_files_in_ram[csv_file] = annotations_arr

        if self.load_in_ram:
            label_distribution = [0, 0, 0]
            length = len(self)
            for i in tqdm(range(length)):
                sample, label = self.get_item(i)
                if label_distribution[2] >= max_silence and label == 2:
                    continue
                label_distribution[label] += 1
                self.samples.append((sample, label))

            print(f"set distribution: {label_distribution}")
        else:
            self.__getitem__(0)

    def __len__(self):
        return round((self.num_audio_files - self.remove_from_set) * self.items_per_clip) if (not self.load_in_ram) or len(self.samples) == 0 else len(self.samples)

    def __getitem__(self, index):
        if self.load_in_ram:
            return self.samples[index]
        else:
            return self.get_item(index)

    def get_item(self, index):
        file_index = math.floor(index / self.items_per_clip)

        audio_sample_path, audio_label_path = self._get_paths(file_index)

        sample_timestamp_begin = (((self.sample_step / self.target_sample_rate) * index) %
                                  (self.clipLength - self.num_samples / self.target_sample_rate))

        _label = self._get_audio_sample_label(audio_label_path, sample_timestamp_begin)

        if self.load_in_ram:
            _signal = self.audio_files_in_ram[audio_sample_path]
        else:
            _signal, sr = torchaudio.load(audio_sample_path)
            _signal = _signal.to(self.device)
            _signal = self._resample_if_necessary(_signal, sr)
            _signal = self._mix_down_if_necessary(_signal)
            _signal = tF.pad(_signal, (0, self.target_sample_rate * self.clipLength - _signal.shape[1]))

        _signal = self._refine_range(_signal, index)

        old_shape = _signal.shape

        if self.transformation is not None:
            _signal = self.transformation(_signal)
            if self.cutoff_point != 0:
                cutoff_index = int((2 * self.cutoff_point / self.target_sample_rate) * _signal.shape[1])
                _signal[0, 0:cutoff_index, :] = 0
        if index == 0:
            self.target_dimension_2 = _signal.shape[2]
        if _signal.shape[2] != self.target_dimension_2:
            _signal = tF.pad(_signal, (0, self.target_dimension_2 - _signal.shape[2]))
            #print(f"index: {index}, old shape: {old_shape}, new shape: {signal.shape}") #debug code

        return _signal, _label

    def _resample_if_necessary(self, _signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            _signal = resampler(_signal).to(self.device)
        return _signal

    def _refine_range(self, _signal, index):
        index = math.floor(index % self.items_per_clip)

        _signal = _signal[:, index * self.sample_step: index * self.sample_step + self.num_samples]
        return _signal

    def _mix_down_if_necessary(self, _signal):
        if _signal.shape[0] > 1:
            _signal = torch.mean(_signal, dim=0, keepdim=True)
        return _signal

    def _get_paths(self, index):
        label_file = self.files[index * 2]
        audio_file = self.files[index * 2 + 1]  # +1 because .wav comes after .csv

        return os.path.join(self.audio_dir, audio_file), os.path.join(self.audio_dir, label_file)

    def _get_audio_sample_label(self, file_name, timestamp_low):

        if self.load_in_ram:
            annotations_arr = self.csv_files_in_ram[file_name]
        else:
            annotations = pd.read_csv(file_name, header=None)
            annotations_arr = annotations.to_numpy().flatten()
            if annotations_arr[len(annotations_arr) - 1] == 0:
                annotations_arr = annotations_arr[0:-4]

        # i % 4 == 0 -> BEGIN_INHALE, i % 4 == 1 -> END_INHALE, == 2 -> BEGIN_EXHALE, ==3 -> END_EXHALE
        timestamp_low *= 1000
        timestamp_high = timestamp_low + (1000 * self.num_samples / self.target_sample_rate)

        for i in range(len(annotations_arr)):
            low_bound, high_bound = annotations_arr[i], self.clipLength * 1000
            if i != len(annotations_arr) - 1:
                high_bound = annotations_arr[i + 1]

            if low_bound <= timestamp_high and timestamp_low <= high_bound:
                # ranges overlap: THIS IS SET FOR 0->breath, 1->no breath, changed final output in cnn too
                match i % 4:
                    case 0:
                        return 0
                    case 2:
                        return 1
                    case _:
                        return 2
        return 2


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    AUDIO_DIRECTORY = "../datasets/breathingSet2/audioMyAnnotations"
    TARGET_SAMPLE_RATE = 16000
    NUM_SAMPLES = 1000
    SAMPLE_STEP = 480

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=2048,
        hop_length=128,
        n_mels=64
    )

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=128
    )

    bs2 = BreathSet2(AUDIO_DIRECTORY, TARGET_SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=spectrogram,
                     _device=device, load_in_ram=True, max_silence=10000, remove_from_set=0)

    print(f"There are {len(bs2)} samples in the dataset.")

    ls = [0, 0, 0]
    # some of them have 2 in their -> based on clip length, some of them are at the end of the clip so its not the full length. can just pad or use samplerate divisible by num samples
    for i in tqdm(range(0, len(bs2))):
        signal, l = bs2[i]
        ls[l] += 1

    print(ls)


    '''plt.figure(figsize=(10, 6))
    plt.imshow(signal.cpu()[0].log2(), aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()'''

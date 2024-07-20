import math
import random
import shutil
from pydub import AudioSegment
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
import time
import cv2

'''
nightmare nightmare nightmare
current dataloader its horrible so bad so bad

workflow:

dataset is found in two places:
datasets/other_sounds
datasets/breathset2

breathset1 is for old stuff
breathset3 has some files from youtube, but ultimately it gets put into bs2

other_sounds is easier and more organised. Number in the folder corresponds to the index of prediction (can be found in predict.py, class mapping)
/dirty/ -> just the raw file of whatever sound. you can clean by taking out silence and other sounds
/train/ -> the train set, will be populated when preprocess is run
/test/ -> the test set, will be populated when preprocess is run

breathSet2 contains a number of folders. 
/audio/ -> where you should put audio that has been downloaded from firebase. 
/temp/ often when I am doing annotations on new data, I put them into this folder instead of doing it from audio, just cuz its easier
/audioMyAnnotations/ and /auduoMyAnnotations so far/ -> used to use, not anymore
/dirtyAudio/ -> where audio files and annotations go when they have been cleaned and annotated 
/cleanAudio/ -> where audio files go after being preprocessed
/cleanAudioTest/ -> move some files from cleanAudio into here so there is a test set


ALSO
there is datasets/cache for the datasets that are ready to be loaded with train.py.

'''

# idea: loop through all wav files and add their samples to one array, then cut that array into one pretdetermined size and save the new wav files
# -> essentially take a number of wave files, combine into one file, then separate into new files again.


# this can do many many things, too many, some of them don't work. see bottom of file for usage
class BreathSetReader(Dataset):
    def __init__(self, audio_dir, sample_rate, samples_per_chunk, chunk_step, transformation, target_clip_length=60,
                 _device="cpu", set_range=(0, -1), cutoff_point=0, other_sets_dir="",
                 load_in_ram=False, max_distribution=(-1, -1, -1, -1, -1), raw_dir="", preprocess=False, cache_dir="",
                 save_to_cache=False, cache_postfix=""):

        # settings:
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.samples_per_chunk = samples_per_chunk
        self.chunk_step = chunk_step
        self.transformation = transformation.to(_device)
        self.clip_length = target_clip_length
        self.device = _device
        self.set_range = set_range
        self.cutoff_point = cutoff_point
        self.load_in_ram = load_in_ram
        self.max_distribution = max_distribution
        self.raw_dir = raw_dir
        self._preprocess = preprocess
        self.cache_dir = cache_dir
        self.save_to_cache = save_to_cache
        self.cache_postfix = cache_postfix
        self.other_dirs = other_sets_dir

        # useful values:
        self.length_of_chunk = 1000 * self.samples_per_chunk / self.sample_rate  #in ms
        self.samples_per_clip = sample_rate * self.clip_length
        self.chunks_per_clip = int((self.samples_per_clip - self.samples_per_chunk) / self.chunk_step) + 1
        self.samples_per_clip = self.chunk_step * (self.chunks_per_clip - 1) + self.samples_per_chunk

        if self.raw_dir != "":
            # pre process audio from raw dir and put into audio dir first:
            self.preprocess()

        self.files = os.listdir(self.audio_dir)
        self.num_audio_files = int(len(self.files) / 2)

        self.final_clip_len = self.final_clip_length()

        self.load_in_ram = load_in_ram

        if self.load_in_ram:
            self.chunks = []
            self.labels = []
            self.load_all()
            if self.save_to_cache:
                self.save_cache()

    def __len__(self):
        if self.load_in_ram:
            return len(self.chunks)

        if self.set_range[1] != -1:
            return self.set_range[1] - self.set_range[0]

        return self.chunks_per_clip * (self.num_audio_files - 1) + self.final_clip_len - self.set_range[0]

    def __getitem__(self, index):
        if self.load_in_ram:
            return self.chunks[index], self.labels[index]
        else:
            return self.get_item(index + self.set_range[0]) if index < self.set_range[1] or self.set_range[
                1] == -1 else None

    def get_item(self, index):
        file_index = math.floor(index / self.chunks_per_clip)

        _signal, sr = torchaudio.load(os.path.join(self.audio_dir, self.files[file_index * 2 + 1]))
        _signal = _signal.to(self.device)
        _signal = self._resample_if_necessary(_signal, sr)
        _signal = self._mix_down_if_necessary(_signal)
        _signal = self._refine_range(_signal, index)

        clip_length = _signal.shape[1] / self.sample_rate  # in seconds
        low_timestamp = ((index * self.chunk_step) / _signal.shape[1]) * 1000 * clip_length
        _label = self._get_audio_sample_label(os.path.join(self.audio_dir, self.files[file_index * 2]), low_timestamp)

        _signal = self.transformation(_signal)
        if self.cutoff_point != 0:
            cutoff_index = int((2 * self.cutoff_point / self.sample_rate) * _signal.shape[1])
            _signal[0, 0:cutoff_index, :] = 0

        return _signal, _label

    def _resample_if_necessary(self, _signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            if len(_signal) == 0:
                print(_signal)
            _signal = resampler(_signal).to(self.device)
        return _signal

    def _refine_range(self, _signal, index):
        index = math.floor(index % self.chunks_per_clip)

        _signal = _signal[:, index * self.chunk_step: index * self.chunk_step + self.samples_per_chunk]
        return _signal

    def _mix_down_if_necessary(self, _signal):
        if _signal.shape[0] > 1:
            _signal = torch.mean(_signal, dim=0, keepdim=True)
        return _signal

    def final_clip_length(self):
        signal, sr = torchaudio.load(os.path.join(self.audio_dir, self.files[len(self.files) - 1]))

        signal = self._resample_if_necessary(signal, sr)

        return int((signal.shape[1] - self.samples_per_chunk) / self.chunk_step) + 1

    #ts low is ms
    def _get_audio_sample_label(self, file_name, timestamp_low):
        annotations = pd.read_csv(file_name, header=None)
        annotations_arr = annotations.to_numpy().flatten()
        if annotations_arr[len(annotations_arr) - 1] == 0:
            annotations_arr = annotations_arr[0:-4]

        # i % 4 == 0 -> BEGIN_INHALE, i % 4 == 1 -> END_INHALE, == 2 -> BEGIN_EXHALE, ==3 -> END_EXHALE
        timestamp_high = timestamp_low + self.length_of_chunk
        max_overlap = [2, 0]  # -> label, overlap
        for i in range(len(annotations_arr)):
            low_bound, high_bound = annotations_arr[i], self.clip_length * 1000
            if i != len(annotations_arr) - 1:
                high_bound = annotations_arr[i + 1]

            overlap = min(high_bound, timestamp_high) - max(low_bound.item(), timestamp_low)

            if overlap > max_overlap[1]:
                max_overlap[1] = overlap
                max_overlap[0] = self.index_to_label(i)

            '''if low_bound <= timestamp_high and timestamp_low <= high_bound:
                # ranges overlap: THIS IS SET FOR 0->breath, 1->no breath, changed final output in cnn too
                return self.index_to_label(i)'''
        return max_overlap[0]

    def index_to_label(self, index):
        match index % 4:
            case 0:
                return 0
            case 2:
                return 1
            case _:
                return 2

    def load_all(self):
        label_distribution = [0] * len(self.max_distribution)
        loaded_so_far = 0
        for i in tqdm(range(0, len(self.files), 2)):
            _signal, sr = torchaudio.load(os.path.join(self.audio_dir, self.files[i + 1]))
            _signal = _signal.to(self.device)
            _signal = self._resample_if_necessary(_signal, sr)
            _signal = self._mix_down_if_necessary(_signal)
            clip_length = _signal.shape[1] / self.sample_rate  # in seconds
            for j in range(self.samples_per_chunk, _signal.shape[1], self.chunk_step):
                chunk = _signal[:, j - self.samples_per_chunk:j]

                low_timestamp = ((j - self.samples_per_chunk) / _signal.shape[1]) * 1000 * clip_length
                label = self._get_audio_sample_label(os.path.join(self.audio_dir, self.files[i]), low_timestamp)

                chunk = self.transformation(chunk)
                if self.cutoff_point != 0:
                    cutoff_index = int((2 * self.cutoff_point / self.sample_rate) * _signal.shape[1])
                    _signal[0, 0:cutoff_index, :] = 0

                if label_distribution[label] < self.max_distribution[label] or self.max_distribution[label] < 0:
                    loaded_so_far += 1

                    if loaded_so_far >= self.set_range[0] and (
                            loaded_so_far < self.set_range[1] or self.set_range[1] == -1):
                        label_distribution[label] += 1
                        self.chunks.append(chunk)
                        self.labels.append(label)

                    if loaded_so_far >= self.set_range[1] and self.set_range[1] != -1:
                        break

        # each folder will be labeled with a number
        if self.other_dirs != "":
            folders = os.listdir(self.other_dirs)

            for folder in folders:
                label = int(folder)

                files = os.listdir(os.path.join(self.other_dirs, folder))

                for file in files:
                    _signal, sr = torchaudio.load(os.path.join(self.other_dirs, folder, file))
                    _signal = _signal.to(self.device)
                    _signal = self._resample_if_necessary(_signal, sr)
                    _signal = self._mix_down_if_necessary(_signal)
                    for j in range(self.samples_per_chunk, _signal.shape[1], self.chunk_step):
                        chunk = _signal[:, j - self.samples_per_chunk:j]

                        chunk = self.transformation(chunk)
                        if self.cutoff_point != 0:
                            cutoff_index = int((2 * self.cutoff_point / self.sample_rate) * _signal.shape[1])
                            _signal[0, 0:cutoff_index, :] = 0

                        if label_distribution[label] < self.max_distribution[label] or self.max_distribution[label] < 0:
                            loaded_so_far += 1

                            if loaded_so_far >= self.set_range[0] and (
                                    loaded_so_far < self.set_range[1] or self.set_range[1] == -1):
                                label_distribution[label] += 1
                                self.chunks.append(chunk)
                                self.labels.append(label)

                            if loaded_so_far >= self.set_range[1] and self.set_range[1] != -1:
                                break

        print(f"set distribution: {label_distribution}")

    def save_cache(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        chunks = torch.stack(self.chunks)
        labels = torch.tensor(self.labels)

        torch.save(chunks, os.path.join(self.cache_dir, f"chunks_{self.cache_postfix}.pt"))
        torch.save(labels, os.path.join(self.cache_dir, f"labels_{self.cache_postfix}.pt"))

        print(f"cached to: {self.cache_dir}")

    # [(n samples per clip C) - (n samples per chunk c)] / (step size S) + 1 = int(n chunks per clip n)

    def preprocess(self):
        if not self._preprocess:
            return

        if os.path.exists(self.audio_dir) and self._preprocess:
            # Use shutil.rmtree to delete the directory and its contents
            shutil.rmtree(self.audio_dir)
            print(f"Directory '{self.audio_dir}' has been deleted along with its contents.")
        if self._preprocess:
            os.mkdir(self.audio_dir)

        files = os.listdir(self.raw_dir)

        all_annotations = np.zeros(0)
        current_file_to_write = torch.zeros((1, self.samples_per_clip))
        current_file_to_write_index = 0
        total_samples_read = 0
        total_samples_written = 0
        total_annotation_indexes = 0
        for i in range(0, len(files), 2):
            annotations = pd.read_csv(os.path.join(self.raw_dir, files[i]), header=None)
            annotations_arr = annotations.to_numpy().flatten()
            if annotations_arr[len(annotations_arr) - 1] == 0:
                annotations_arr = annotations_arr[0:-4]

            signal, sample_rate = torchaudio.load(os.path.join(self.raw_dir, files[i + 1]))
            signal = signal.to(self.device)
            signal = self._mix_down_if_necessary(signal)
            signal = self._resample_if_necessary(signal, sample_rate)

            annotations_arr = (annotations_arr / (1000 * (signal.shape[1] / self.sample_rate))) * signal.shape[
                1]  # annotations_arr now in samples of current file

            all_annotations = np.append(all_annotations, annotations_arr + total_samples_read)

            signal_read_index = 0
            while signal_read_index < signal.shape[1] - 1:
                length_to_write = min(self.samples_per_clip - current_file_to_write_index,
                                      signal.shape[1] - signal_read_index)
                current_file_to_write[:,
                current_file_to_write_index: current_file_to_write_index + length_to_write] = signal[:,
                                                                                              signal_read_index: signal_read_index + length_to_write]

                current_file_to_write_index += length_to_write
                signal_read_index += length_to_write

                if current_file_to_write_index >= self.samples_per_clip:
                    file_annotations = (all_annotations[(all_annotations >= total_samples_written) & (
                            all_annotations < total_samples_written + self.samples_per_clip)]
                                        % self.samples_per_clip)

                    left_pad = total_annotation_indexes % 4
                    total_annotation_indexes += len(file_annotations)
                    right_pad = 4 - total_annotation_indexes % 4 if total_annotation_indexes % 4 != 0 else 0
                    file_annotations = (file_annotations / self.sample_rate) * 1000
                    file_annotations = np.pad(file_annotations, pad_width=(left_pad, right_pad), mode='constant',
                                              constant_values=(-1, self.clip_length * 1000 + 1))
                    df = pd.DataFrame(file_annotations.reshape(1, -1).astype(np.int64))
                    df.to_csv(os.path.join(self.audio_dir, f"audiofile_{total_annotation_indexes:04}.csv"), index=False,
                              header=False)

                    torchaudio.save(os.path.join(self.audio_dir, f"audiofile_{total_annotation_indexes:04}.wav"),
                                    current_file_to_write, self.sample_rate)
                    total_samples_written += self.samples_per_clip
                    current_file_to_write_index = 0

            total_samples_read += signal.shape[1]

        file_annotations = (all_annotations[(all_annotations >= total_samples_written) & (
                all_annotations < total_samples_written + self.samples_per_clip)]
                            % self.samples_per_clip)
        # left padding:
        left_pad = total_annotation_indexes % 4
        total_annotation_indexes += len(file_annotations)
        right_pad = 4 - total_annotation_indexes % 4 if total_annotation_indexes % 4 != 0 else 0
        file_annotations = (file_annotations / self.sample_rate) * 1000
        file_annotations = np.pad(file_annotations, pad_width=(left_pad, right_pad), mode='constant',
                                  constant_values=(-1, self.clip_length * 1000 + 1))

        df = pd.DataFrame(file_annotations.reshape(1, -1).astype(np.int64))
        df.to_csv(os.path.join(self.audio_dir, f"Zaudio.csv"), index=False, header=False)

        torchaudio.save(os.path.join(self.audio_dir, f"Zaudio.wav"),
                        current_file_to_write[:, 0:current_file_to_write_index], self.sample_rate)
        total_samples_written += self.samples_per_clip


class CachedDataSetReader(Dataset):
    def __init__(self, cache_dir="", cache_postfix="", max_dist=(0, 0, 0)):
        self.cache_dir = cache_dir
        self.cache_postfix = cache_postfix
        self.max_dist = max_dist

        self.chunks = torch.empty(0)
        self.labels = torch.empty(0)

        self.load_cache()

        if self.max_dist != (0, 0, 0):
            self.limit_distribution()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.chunks[index], self.labels[index]

    def load_cache(self):
        if not os.path.exists(self.cache_dir):
            return

        self.chunks = torch.load(os.path.join(self.cache_dir, f"chunks_{self.cache_postfix}.pt"))
        self.labels = torch.load(os.path.join(self.cache_dir, f"labels_{self.cache_postfix}.pt"))
        print(f"loaded chunks and labels from {self.cache_dir}")

    def limit_distribution(self):
        length = np.sum(self.max_dist)
        new_chunks = torch.zeros((length, self.chunks.shape[1], self.chunks.shape[2], self.chunks.shape[3]))
        new_labels = torch.zeros(length)

        distribution = [0] * len(self.max_dist)
        top_pointer = 0
        for i in tqdm(range(len(self.labels))):
            lbl = self.labels[i]
            if distribution[lbl] < self.max_dist[lbl]:
                new_labels[top_pointer] = lbl
                new_chunks[top_pointer] = self.chunks[i]
                top_pointer += 1
                distribution[lbl] += 1

        self.chunks = new_chunks
        self.labels = new_labels.type(torch.int64)
        print(f"set distribution: {distribution}")


def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')
    os.remove(mp3_path)


# Usage
def preprocess_other(root_folder, target_folder, _device, max_length_s=300000000, do_not_save=set()):
    folders = os.listdir(root_folder)

    for folder in folders:
        current_length = 0

        files = os.listdir(os.path.join(root_folder, folder))
        random.shuffle(files)

        if os.path.exists(os.path.join(target_folder, folder)):
            shutil.rmtree(os.path.join(target_folder, folder))
        os.mkdir(os.path.join(target_folder, folder))

        for file in files:
            if ".mp3" in file:
                convert_mp3_to_wav(os.path.join(root_folder, folder, file), os.path.join(root_folder, folder, file.replace(".mp3", ".wav")))
                file = file.replace(".mp3", ".wav")

            path = os.path.join(root_folder, folder, file)
            if path in do_not_save:
                continue

            do_not_save.add(path)

            _signal, sr = torchaudio.load(path)

            _signal = _signal.to(_device)
            # Create a Vad transform
            vad = torchaudio.transforms.Vad(sample_rate=sr, trigger_time=0.01, allowed_gap=0, trigger_level=7).to(_device)

            mean = (_signal**2).mean().sqrt()
            threshold = mean / 2

            mask = torch.empty(_signal.shape, dtype=torch.bool)

            for i in range(0, _signal.shape[1], int(sr / 3)):
                if (_signal[0, i:i + int(sr/3)]**2).mean().sqrt() > threshold:
                    mask[0, i:i + int(sr/3)] = True

            _signal = _signal[mask]

            # Apply the transform to your audio waveform
            # _signal = vad(_signal)
            _signal = _signal.reshape(1, -1)
            current_length += _signal.shape[1] / sr

            if current_length > max_length_s:
                break

            torchaudio.save(os.path.join(target_folder, folder, file), _signal.cpu(), sr)



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=4096,
        hop_length=128,
        n_mels=64
    )

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=128
    )

    # when you add new breath audio data that has been annotated, you need to uncomment this and run it. it will take the audio from
    # raw_dir and condense it and put it into cleanAudio. Relevant settins are target_clip_length, which will make the clean audio clips this length in seconds.
    ''' BreathSetReader("../datasets/breathingSet2/cleanAudio", 16000, 8000, 1000, mel_spectrogram,
                                  target_clip_length=3 * 60, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device,
                                  load_in_ram=False, preprocess=True, other_sets_dir="../datasets/other_sounds/test",
                                  save_to_cache=False, cache_dir="../datasets/cache", cache_postfix="mel_8000_1000_test")'''
    dont_save = set()
    # this preprocesses the 'other' sounds, uncomment and run when there is new 'other' sounds.
    # preprocess_other("../datasets/other_sounds/dirty", "../datasets/other_sounds/test", _device=device, max_length_s=150, do_not_save=dont_save)

    #any time you want to generate a new cached dataset, run this. this is when sample rate, samples per chunk, chunk step and mel_spectrogtam matter.
    # do not run thi sand the initial preprocess (two functions up) in the same run because you need to go in by hand and put some files from /networks/breathSet2/cleanAudio into /networks/breathSet2/cleanAudioTest
    BreathSetReader("../datasets/breathingSet2/cleanAudioTest", 16000, 4096, 1000, mel_spectrogram,
                              target_clip_length=3 * 60, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device,
                              load_in_ram=True, preprocess=False, other_sets_dir="../datasets/other_sounds/test",
                              save_to_cache=True, cache_dir="../datasets/cache", cache_postfix="mel_4096_1000_test")

    # do above again but for the test set:
    # preprocess_other("../datasets/other_sounds/dirty", "../datasets/other_sounds/train", _device=device, max_length_s=1000, do_not_save=dont_save)

    BreathSetReader("../datasets/breathingSet2/cleanAudio", 16000, 4096, 1000, mel_spectrogram,
                              target_clip_length=3 * 60, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device,
                              load_in_ram=True, preprocess=False, other_sets_dir="../datasets/other_sounds/train",
                              save_to_cache=True, cache_dir="../datasets/cache", cache_postfix="mel_4096_1000_train") # 4096 uses a different mel_spectrogram, where n_fft=samplesperchunk=4096


    # this is just old i think:
    '''dataset = BreathSetReader("../datasets/breathingSet2/cleanAudioTest", 16000, 4000, 500, mel_spectrogram,
                              target_clip_length=3 * 60, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device,
                              load_in_ram=True, preprocess=False,
                              load_from_cache=True, cache_dir="../datasets/cache", cache_postfix="mel_4000_500_test")

    print(f"there are: {len(dataset)} elements in the dataset")

    dist = [0, 0, 0]
    for i in tqdm(range(0, len(dataset))):
        sgnl, lbl = dataset[i]
        dist[lbl] += 1'''
    '''
        img = sgnl.cpu()[0].numpy()

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
            continue


         plt.figure(figsize=(10, 6))
        plt.imshow(sgnl.cpu()[0].log2(), aspect='auto', origin='lower')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
        '''

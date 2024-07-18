from tqdm import tqdm

from custom_dataset import BreathSet2
from breath_data_reader import BreathSetReader
import numpy as np
import torchaudio
import torch
import cv2

AUDIO_DIR = "../datasets/breathingSet2/cleanAudio"
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = 2000
CHUNK_STEP = 500


def show_spectrogram(spectrogram, lbl, name):
    img = spectrogram.cpu()[0].numpy()

    np_img = np.flipud(img)
    np_img = cv2.resize(np_img, (img.shape[1] * 18, img.shape[0] * 6))

    old_min, old_max = np.min(np_img), np.max(np_img)
    new_min, new_max = 0, 1

    if old_min != old_max:
        np_img = ((np_img - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0.5, 0.5, 0.5)  # white
    position = (50, 50)

    # Add the text
    cv2.putText(np_img, str(lbl), position, font, font_scale, color)

    cv2.imshow(name, np_img)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=128,
        n_mels=64
    )

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=128
    )

    old_set = BreathSet2(AUDIO_DIR, SAMPLE_RATE, SAMPLES_PER_CHUNK, CHUNK_STEP, transformation=mel_spectrogram,
                         _device=device, load_in_ram=True, remove_from_set=3)

    new_set = BreathSetReader(AUDIO_DIR, SAMPLE_RATE, SAMPLES_PER_CHUNK, CHUNK_STEP,
                              mel_spectrogram,
                              target_clip_length=60, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device,
                              load_in_ram=True, preprocess=False, set_range=(0, 24909))

    print(f"old set length: {len(old_set)}")
    print(f"new set length: {len(new_set)}")

    for i in tqdm(range(0, len(old_set))):
        old_signal, old_lbl = old_set[i]
        if i >= len(new_set):
            print("over")
        else:
            new_signal, new_lbl = new_set[i]

            if new_lbl != old_lbl or not new_signal.equal(old_signal):
                print(f"{i}: {old_lbl}, {new_lbl}")
            '''show_spectrogram(new_signal, new_lbl, "new signal")'''

        '''show_spectrogram(old_signal, old_lbl, "old signal")
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        else:
            continue'''

    cv2.destroyAllWindows()
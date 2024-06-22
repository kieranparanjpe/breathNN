import math
import cv2
from new_set import cnnB
from new_set import predictB
import numpy as np
import pyaudio
import time
import wave
import csv
import torchaudio
import torch
from collections import deque

FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 16000
NUM_SAMPLES = 4000
SAMPLE_STEP_SIZE = 480
INPUT_DEVICE = 1

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=2048,
    hop_length=128,
    n_mels=64
)


class Microphone:

    def __init__(self):
        self.stream = None
        self.audio = pyaudio.PyAudio()
        self.frames = deque()
        self.start_time = -1

        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        for i in range(0, num_devices):
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                      self.audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    def start(self):
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=SAMPLE_RATE,
                                      input=True,
                                      stream_callback=self.mic_callback,
                                      frames_per_buffer=SAMPLE_STEP_SIZE,
                                      input_device_index=INPUT_DEVICE)
        self.start_time = time.time()

    def mic_callback(self, input_data, frame_count, time_info, flags):
        audio_data = np.frombuffer(input_data, dtype=np.float32)

        for data in audio_data:
            self.frames.append(data)

        while len(self.frames) > NUM_SAMPLES:
            self.frames.popleft()

        #if len(self.frames) > NUM_SAMPLES:

        return None, pyaudio.paContinue

    def close(self):
        if self.start_time < 0:
            return

        if self.stream is not None:
            self.stream.close()
        self.audio.terminate()


if __name__ == '__main__':
    cnn = cnnB.CNNNetwork()
    state_dict = torch.load("new_set\\networks\\feedforwardnet1719038075.pth")
    cnn.load_state_dict(state_dict)

    microphone = Microphone()
    microphone.start()

    last_stable_prediction = "other"
    prediction_buffer = deque()
    prediction_buffer.append(last_stable_prediction)

    while True:
        if len(microphone.frames) == NUM_SAMPLES:
            tensor = torch.tensor(microphone.frames)
            spectrogram = mel_spectrogram(tensor)
            spectrogram_img = cv2.resize(np.flipud(spectrogram.numpy()), (1024, 512))
            spectrogram.unsqueeze_(0)
            spectrogram.unsqueeze_(0)

            prediction, expected = predictB.predict(cnn, spectrogram, 0,
                                          predictB.class_mapping)

            prediction_text = last_stable_prediction

            if prediction_buffer[0] != prediction:
                prediction_buffer.clear()

            prediction_buffer.append(prediction)

            if len(prediction_buffer) > 3:
                last_stable_prediction = prediction
                prediction_text = prediction

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0.5, 0.5, 0.5)  # white
            position = (50, 50)

            # Add the text
            cv2.putText(spectrogram_img, str(prediction_text), position, font, font_scale, color)
            cv2.imshow("spectrogram", spectrogram_img)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    microphone.close()
    cv2.destroyAllWindows()

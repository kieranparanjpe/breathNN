import cv2
#import cnnProd as cnn
import cnn
import predict
import numpy as np
import pyaudio
import time
import torchaudio
import torch
from collections import deque

'''
Purpose of file:
Test out a model in realtime
'''

# basic params about network
FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 16000
NUM_SAMPLES = 8000
SAMPLE_STEP_SIZE = 250
BUFFER_SIZE = 4
INPUT_DEVICE = 1
LOAD_NETWORK = "feedforwardnet234.pth"
frame_processed = False

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

        self.last_stable_prediction = ("silence", 0)
        self.prediction_buffer = deque()
        self.prediction_buffer.append(self.last_stable_prediction)

        self.square_amplitude_sum = 0
        self.total_frames_processed = 0

        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        for i in range(0, num_devices):
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                      self.audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    def start(self):
        # starts mic stream on a thread, will call mic_callback with audio data each time it is available
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=SAMPLE_RATE,
                                      input=True,
                                      stream_callback=self.mic_callback,
                                      frames_per_buffer=SAMPLE_STEP_SIZE,
                                      input_device_index=INPUT_DEVICE)
        self.start_time = time.time()

    # gets called each time mic data is available
    def mic_callback(self, input_data, frame_count, time_info, flags):
        global frame_processed
        # get input data in useful format
        audio_data = np.frombuffer(input_data, dtype=np.float32)

        # frames is a queue, we are enqueuing the data
        for data in audio_data:
            self.frames.append(data)

        # if we have enqueued too much data, we will deQ some.
        while len(self.frames) > NUM_SAMPLES:
            self.frames.popleft()

        #use a Q because: we have a chunk size (NUM_SAMPLES) which will be say 0.25 seconds or 0.5 seconds, but we want to do an inference more frequently
        #say for example eacho 0.1 seconds, so we add 0.1 seconds worth of samples to the q each time, and remove 0.1 seconds. keeps it smooth :)

        #if len(self.frames) > NUM_SAMPLES:
        frame_processed = False

        return None, pyaudio.paContinue

    # close stream
    def close(self):
        if self.start_time < 0:
            return

        if self.stream is not None:
            self.stream.close()
        self.audio.terminate()

    # used for doing prediction
    def predict(self):
        if len(microphone.frames) == NUM_SAMPLES:
            # pre processing:
            frames = torch.tensor(microphone.frames)

            spectrogram = mel_spectrogram(frames)
            spectrogram_img = cv2.resize(np.flipud(spectrogram.numpy()), (1024, 512))
            spectrogram.unsqueeze_(0)
            spectrogram.unsqueeze_(0)

            # calculte root mean square of signal, and also rolling rms of the signal, +0.005 for some reason, was messing with values IG.
            rmsAmplitude = (spectrogram.abs()).mean().item()
            rolling = (self.square_amplitude_sum / (self.total_frames_processed if self.total_frames_processed != 0 else 1))+0.005
            print(f"rmsAmplitude: {rmsAmplitude}\nrolling: {rolling}")

            '''if rmsAmplitude <= rolling:
                prediction = "too low"
                certainty = torch.tensor(rmsAmplitude)
            else:'''
            # do prediction
            prediction, certainty = predict.predict(cnn, spectrogram, threshold=0.9)


            # next bit is just playing with activation thresholds, and drawing two squares to the screen. height of one is rms, height of other is rolling rms. not in use rn i think.
            scale_factor = -2000
            reset_after = 48

            cv2.rectangle(spectrogram_img, (200, 200 + int(rmsAmplitude * scale_factor)), (250, 250 + int(rmsAmplitude * scale_factor)),
                          255, 1)
            cv2.rectangle(spectrogram_img, (300, 200 + int(rolling * scale_factor)), (350, 250 + int(rolling * scale_factor)),
                          255, 1)

            self.square_amplitude_sum += spectrogram.abs().sum().item()
            self.total_frames_processed += NUM_SAMPLES

            if self.total_frames_processed >= NUM_SAMPLES * reset_after:
                self.total_frames_processed = 0
                self.square_amplitude_sum = 0

            if prediction is None:
                prediction = "No prediction"

            # show label text
            prediction_text = f"{self.last_stable_prediction[0]} {certainty.item():.2f}"


            # prediction buffer is to for guess stabilization (you need BUFFER_SIZE number of the same guess in a row for it to update)
            if self.prediction_buffer[0][0] != prediction:
                self.prediction_buffer.clear()

            self.prediction_buffer.append((prediction, certainty))

            if len(self.prediction_buffer) > BUFFER_SIZE:
                self.last_stable_prediction = prediction, certainty
                prediction_text = f"{prediction} {certainty.item():.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0.5, 0.5, 0.5)  # white
            position = (50, 50)

            # Add the text
            cv2.putText(spectrogram_img, str(prediction_text), position, font, font_scale, color)
            cv2.imshow("spectrogram", spectrogram_img)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # load network
    cnn = cnn.CNNNetwork(num_outputs=5).to(device)
    state_dict = torch.load(f"../networks/{LOAD_NETWORK}", map_location=torch.device(device))
    cnn.load_state_dict(state_dict)
    cnn = cnn.to(device)

    microphone = Microphone()
    microphone.start()

    # main loop
    while True:
        if not frame_processed:
            microphone.predict()
            frame_processed = True

        # q to close window
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    microphone.close()
    cv2.destroyAllWindows()

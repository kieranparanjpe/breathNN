import torch
import torchaudio
from tqdm import tqdm
if __name__ == "__main__":
    from cnnB import CNNNetwork
    from dataset2 import BreathSet2

if __name__ == '__main__':
    from trainB import AUDIO_DIR, SAMPLE_STEP, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "inhale",
    "exhale",
    "other",
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    #state_dict = torch.load("feedforwardnet.pth")
    #cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=128,
        n_mels=64
    )
    test_set = BreathSet2(AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=mel_spectrogram,
                           _device="cpu", load_in_ram=True)

    #1292
    # get a sample from the urban sound dataset for inference

    correct = 0
    total = 0
    nonOther = 0
    for i in tqdm(range(len(test_set))):
        input, target = test_set[i]  # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        # make an inference
        predicted, expected = predict(cnn, input, target,
                                      class_mapping)
        total += 1
        if predicted == expected:
            correct += 1
        if target != 2:
            nonOther += 1

    print(f"Accuracy: {100 * correct / total} with {nonOther} non others")

import torch
import torchaudio
from tqdm import tqdm

if __name__ == "__main__":
    from cnn import CNNNetwork
    from custom_dataset import BreathSet2
    from train import AUDIO_DIR, SAMPLE_STEP, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "inhale",
    "exhale",
    "other",
]

LOAD_NETWORK = "feedforwardnet1719045554.pth"


def predict(model, _input, _target=None, _class_mapping=None):
    if _class_mapping is None:
        _class_mapping = class_mapping

    model.eval()
    certainty = 0
    with torch.no_grad():
        predictions = model(_input)
        predicted_index = predictions[0].argmax(0)
        certainty = predictions[0][predicted_index]
        p = _class_mapping[predicted_index]
        if _target is not None:
            e = _class_mapping[_target]
    if _target is not None:
        return p, e
    return p, certainty


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load(f"../networks/{LOAD_NETWORK}")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=128,
        n_mels=64
    )
    test_set = BreathSet2(AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=mel_spectrogram,
                          _device="cpu", load_in_ram=False)

    correct = 0
    total = 0
    nonOther = 0
    for i in tqdm(range(len(test_set))):
        _input, target = test_set[i]
        _input.unsqueeze_(0)

        # make an inference
        predicted, expected = predict(cnn, _input, target,
                                      class_mapping)
        total += 1
        if predicted == expected:
            correct += 1
        if target != 2:
            nonOther += 1

    print(f"Accuracy: {100 * correct / total} with {nonOther} non others")

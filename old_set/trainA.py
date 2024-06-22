import csv

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    from dataset import BreathSet1
    import predictA

from cnnA import CNNNetwork

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.0001

ANNOTATIONS_FILE = "/breathingSet1/annotations.csv"
AUDIO_DIR = "/breathingSet1"
SAMPLE_RATE = 44100
NUM_SAMPLES = 4096


def write_file(name, row):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    correct = 0
    total = 0
    loss_sum = 0
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        # calculate loss

        prediction = model(input)
        loss = loss_fn(prediction, target)
        for i in range(len(prediction)):
            total += 1
            if torch.argmax(prediction[i]) == target[i]:
                correct += 1

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_sum += loss.item()

    print(f"loss: {loss_sum / total}, accuracy: {100 * correct / total}")
    return loss_sum / total, 100 * correct / total


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        loss, stat = train_single_epoch(model, data_loader, loss_fn, optimiser, device)

        predict_accuracy = 0
        if i % 20 == 0:
            correct = 0
            total = 0
            print("Verifying model:\n")
            for j in tqdm(range(1292, len(bs2))):
                input, target = bs2[j]  # [batch size, num_channels, fr, time]
                input.unsqueeze_(0)

                # make an inference
                predicted, expected = predict.predict(model, input, target,
                                                      predict.class_mapping)
                total += 1
                if predicted == expected:
                    correct += 1

            predict_accuracy = 100 * correct / total
            print(f"Accuracy: {100 * correct / total}")

        write_file("C:\\Users\\kiera\\Documents\\verse\\breathNN\\stats.csv",
                   [i, stat, loss, predict_accuracy] if predict_accuracy != 0 else [i, stat, loss])
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
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
                     device, 20, 4)

    bs2 = BreathSet1(ANNOTATIONS_FILE,
                     AUDIO_DIR,
                     mel_spectrogram,
                     SAMPLE_RATE,
                     NUM_SAMPLES,
                     device, 20, 0)

    train_dataloader = create_data_loader(bs1, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print(f"Trained feed forward net saved at feedforwardnet.pth with len {len(bs1)}")

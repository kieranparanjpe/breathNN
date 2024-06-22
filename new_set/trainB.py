import csv
import time
from matplotlib import pyplot as plt
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    from dataset2 import BreathSet2
    import predictB

from cnnB import CNNNetwork

BATCH_SIZE = 8
EPOCHS = 21
LEARNING_RATE = 0.0001

AUDIO_DIR = "../datasets/breathingSet2\\audioMyAnnotations"
SAMPLE_RATE = 16000
NUM_SAMPLES = 4000
SAMPLE_STEP = 480

TIME = int(time.time())


def write_file(name, row):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    correct = 0
    total = 0
    loss_sum = 0
    guess_distribution = [0, 0, 0]
    label_distribution = [0, 0, 0]
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        # calculate loss

        prediction = model(input)
        loss = loss_fn(prediction, target)

        for i in range(len(prediction)):
            total += 1
            guess_distribution[prediction[i].argmax()] += 1
            label_distribution[target[i]] += 1
            if torch.argmax(prediction[i]) == target[i]:
                correct += 1

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_sum += loss.item()

    print(f"loss: {loss_sum / total}, accuracy: {100 * correct / total}, distribution_predict: {guess_distribution}, distribution_label: {label_distribution}")
    return loss_sum / total, 100 * correct / total


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    predict_accuracy = 0
    train_accuracy = []
    test_accuracy = []
    costs = []
    for i in range(epochs):
        print(f"Epoch {i}")
        loss, stat = train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        if i % 10 == 0:
            correct = 0
            total = 0
            print("Verifying model:\n")
            for j in tqdm(range(len(train_set), len(test_set))):
                input, target = test_set[j]  # [batch size, num_channels, fr, time]
                if len(input.shape) < 4:
                    input.unsqueeze_(0)

                # make an inference
                predicted, expected = predictB.predict(model, input, target,
                                                       predictB.class_mapping)
                total += 1
                if predicted == expected:
                    correct += 1

            predict_accuracy = 100 * correct / total
            print(f"Accuracy: {100 * correct / total}")

        write_file(f"C:\\Users\\kiera\\Documents\\verse\\breathNN\\new_set\\stats\\stats{TIME}.csv",
                   [i, stat, loss, predict_accuracy])  # if predict_accuracy != 0 else [i, stat, loss])
        train_accuracy.append(stat)
        test_accuracy.append(predict_accuracy)
        costs.append(loss)
        print("---------------------------")

    plt.plot(train_accuracy, label='train accuracy')
    plt.plot(test_accuracy, label='test accuracy')
    plt.show()
    plt.plot(costs, label='cost')
    plt.show()


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
        hop_length=128,
        n_mels=64
    )
    train_set = BreathSet2(AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=mel_spectrogram,
                           _device=device, remove_from_set=3, load_in_ram=True, max_silence=8500)
    #len with 3 removes is 14340
    test_set = BreathSet2(AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=mel_spectrogram,
                          _device=device, load_in_ram=True)

    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    # construct model and assign it to device
    cnn = CNNNetwork().to(device)

    #this is for loading a already trained model to continue training:
    state_dict = torch.load("networks\\feedforwardnet1719038075.pth")
    cnn.load_state_dict(state_dict)

    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)


    # save model
    torch.save(cnn.state_dict(), f"networks\\feedforwardnet{TIME}.pth")
    print(f"Trained feed forward net saved at feedforwardnet{TIME}.pth with len {len(cnn.state_dict())}")

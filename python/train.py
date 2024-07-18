import csv
import os
import shutil
import time
from matplotlib import pyplot as plt
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn import CNNNetwork
from python.breath_data_reader import BreathSetReader, CachedDataSetReader
import numpy as np

if __name__ == '__main__':
    from custom_dataset import BreathSet2
    import predict

LOAD_NETWORK = ""
AUDIO_DIR_TRAIN = "../datasets/breathingSet2/cleanAudio"
AUDIO_DIR_TEST = "../datasets/breathingSet2/cleanAudioTest"
TIME = int(time.time())
EPOCHS = 251

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SAMPLE_RATE = 16000
NUM_SAMPLES = 4000
SAMPLE_STEP = 250
NUM_OUTPUTS = 5


# things to try: reduce complexity and mess around with num samples and sample step
# look at: https://www.youtube.com/watch?v=uCGROOUO_wY
# also confusion matrix
# get other datasets!


def write_file(name, row):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def train_single_epoch(model, data_loader, _loss_fn, _optimiser, _device):
    correct = 0
    correct_breath = 0
    total = 0
    loss_sum = 0
    guess_distribution = [0] * NUM_OUTPUTS
    label_distribution = [0] * NUM_OUTPUTS
    confusion_matrix = np.zeros((NUM_OUTPUTS, NUM_OUTPUTS), dtype=np.float32)  # [guess][label]
    for _input, target in tqdm(data_loader):
        _input, target = _input.to(_device), target.to(_device)

        # calculate loss

        prediction = model(_input)
        loss = _loss_fn(prediction, target)

        for i in range(len(prediction)):
            total += 1
            guess_distribution[prediction[i].argmax()] += 1
            label_distribution[target[i]] += 1
            p = torch.argmax(prediction[i])
            l = target[i]
            if p == l:
                correct += 1

            if ((l == 0 or l == 1) and (p == 0 or p == 1)) or (l >= 2 and p >= 2):
                correct_breath += 1

            confusion_matrix[p, l] += 1

        # backpropagate error and update weights
        _optimiser.zero_grad()
        loss.backward()
        _optimiser.step()
        loss_sum += loss.item()

    print(
        f"loss: {loss_sum / total}, accuracy: {100 * correct / total} breath accuracy: {100 * correct_breath / total}\ndistribution_predict: {guess_distribution}, distribution_label: {label_distribution}")
    print(f"confusion: \n{100 * confusion_matrix / total}")
    return loss_sum / total, 100 * correct / total, 100 * correct_breath / total


def train(model, data_loader, _loss_fn, _optimiser, _device, epochs):
    predict_accuracy = 0
    predict_breath_accuracy = 0
    train_accuracy = []
    test_accuracy = []
    breath_accuracy_train = []
    breath_accuracy_test = []
    costs = []
    shutil.rmtree(f"../networks/temp")
    os.mkdir(f"../networks/temp")
    for i in range(epochs):
        print(f"Epoch {i}")
        loss, stat, breath_stat = train_single_epoch(model, data_loader, _loss_fn, _optimiser, _device)
        if i % 3 == 0:
            correct = 0
            correct_breath = 0
            total = 0
            confusion_matrix = np.zeros((NUM_OUTPUTS, NUM_OUTPUTS), dtype=np.float32)
            print("Verifying model:\n")
            for j in tqdm(range(len(test_set))):
                _input, target = test_set[j]  # [batch size, num_channels, fr, time]
                if len(_input.shape) < 4:
                    _input.unsqueeze_(0)

                # make an inference
                predicted, expected = predict.predict(model, _input, target, predict.class_mapping)
                total += 1
                if predicted == expected:
                    correct += 1
                if ((predicted == "inhale" or predicted == "exhale") and (expected == "inhale" or expected == "exhale")) or predicted == expected:
                    correct_breath += 1

                confusion_matrix[predict.class_mapping.index(predicted), predict.class_mapping.index(expected)] += 1

            predict_accuracy = 100 * correct / total
            predict_breath_accuracy = 100 * correct_breath / total
            print(f"Accuracy: {predict_accuracy}, breath accuracy: {predict_breath_accuracy}")
            print(f"total guesses: {total}")
            print(f"Confusion: \n{confusion_matrix / total * 100}")
            torch.save(cnn.state_dict(), f"../networks/temp/feedforwardnet{i}.pth")

        write_file(f"..\\output\\stats{TIME}.csv", [i, stat, breath_stat, loss, predict_accuracy, predict_breath_accuracy])

        train_accuracy.append(stat)
        test_accuracy.append(predict_accuracy)
        breath_accuracy_test.append(predict_breath_accuracy)
        breath_accuracy_train.append(breath_stat)
        costs.append(loss)
        print("---------------------------")

    plt.plot(train_accuracy, label='train accuracy')
    plt.plot(test_accuracy, label='test accuracy')
    plt.plot(breath_accuracy_train, label='train accuracy breath')
    plt.plot(breath_accuracy_test, label='test accuracy breath')

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

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=128
    )


    '''train_set = BreathSet2(AUDIO_DIR_TRAIN, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=spectrogram,
                           _device=device, load_in_ram=True, max_silence=18000, cutoff_point=300)
    test_set = BreathSet2(AUDIO_DIR_TEST, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, transformation=spectrogram,
                          _device=device, load_in_ram=True, cutoff_point=300)'''

    '''train_set = BreathSetReader(AUDIO_DIR_TRAIN, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, mel_spectrogram,
                                target_clip_length=60*3, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device, load_in_ram=True, preprocess=False,
                                max_distribution=(26000, 26000, 26000), load_from_cache=True, cache_dir="../datasets/cache", cache_postfix="mel_4000_250_train")

    test_set = BreathSetReader(AUDIO_DIR_TEST, SAMPLE_RATE, NUM_SAMPLES, SAMPLE_STEP, mel_spectrogram,
                               target_clip_length=60*3, raw_dir="../datasets/breathingSet2/dirtyAudio", _device=device,
                               load_in_ram=True, preprocess=False, load_from_cache=True, cache_dir="../datasets/cache", cache_postfix="mel_4000_250_test")'''

    train_set = CachedDataSetReader(cache_dir="../datasets/cache", cache_postfix="mel_4096_1000_train", max_dist=(8500, 8500, 8500, 8500, 8500))
    test_set = CachedDataSetReader(cache_dir="../datasets/cache", cache_postfix="mel_4096_1000_test")

    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    # construct model and assign it to device
    cnn = CNNNetwork(num_outputs=5).to(device)

    # this is for loading a already trained model to continue training:
    if LOAD_NETWORK != "":
        print(f"loading: {LOAD_NETWORK}")
        state_dict = torch.load(f"../networks/{LOAD_NETWORK}")
        cnn.load_state_dict(state_dict)

    print(cnn)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), f"../networks/feedforwardnet{TIME}.pth")
    print(f"Trained feed forward net saved at feedforwardnet{TIME}.pth with len {len(cnn.state_dict())}")

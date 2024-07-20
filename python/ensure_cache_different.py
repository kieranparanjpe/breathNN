from tqdm import tqdm

from custom_dataset import BreathSet2
from breath_data_reader import BreathSetReader, CachedDataSetReader
import numpy as np
import torchaudio
import torch
import cv2

'''
script is to double check that the test and train cached sets are actually different
I got higher than expected results once and needed to check
'''

train_set = CachedDataSetReader(cache_dir="../datasets/cache", cache_postfix="mel_8000_1000_train")
test_set = CachedDataSetReader(cache_dir="../datasets/cache", cache_postfix="mel_8000_1000_test")

test_set_set = set()

test_set_set.add((str(train_set[0][0].tolist()), train_set[0][1].item()))
print((str(train_set[0][0].tolist()), train_set[0][1].item()) in test_set_set)

for i in tqdm(range(len(test_set))):
    sgnl, lbl = test_set[i]

    test_set_set.add((str(sgnl.tolist()), lbl.item()))

for i in tqdm(range(len(train_set))):
    sgnl, lbl = train_set[i]

    if (str(sgnl.tolist()), lbl.item()) in test_set_set:
        print(i)

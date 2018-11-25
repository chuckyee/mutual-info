#!/usr/bin/env python

from __future__ import print_function, division

import random

from tqdm import tqdm
import numpy as np
from torchvision import datasets

from mutual_info import mutual_information
from NPEET import entropy_estimators


def main():
    # datadir = "/Users/chuck-houyee/Developer/datasets/fashion-mnist"
    datadir = "/Users/chuck-houyee/Developer/datasets/mnist"
    dataset = datasets.MNIST(datadir, train=True)

    X = []
    Y = []
    N = 1000
    indices = random.sample(range(len(dataset)), N)
    for i in tqdm(indices):
        image, label = dataset[i]
        image = np.asarray(image)
        binary = (image >= 128).astype(float)
        top = binary[:14].flatten()
        bottom = binary[14:].flatten()
        X.append(top)
        Y.append(bottom)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # entropy = mutual_information((X, Y), k=1)
    entropy = entropy_estimators.mi(X, Y)
    print("Entropy = {}".format(entropy))

if __name__ == '__main__':
    main()

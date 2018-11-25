#!/usr/bin/env python

from __future__ import print_function, division

from collections import defaultdict, Counter

from tqdm import tqdm
import numpy as np
from torchvision import datasets


def entanglement_entropy(states):
    # computes entanglement entropy of even superposition of all states
    by_a = defaultdict(list)
    by_b = defaultdict(list)
    for a, b in tqdm(states):
        by_a[a].add(b)
        by_b[b].add(a)
    N = len(states)

def entanglement_entropy(mnist):
    # horizontal cut across middle
    ddict = defaultdict(list)
    for n in tqdm(range(len(mnist))):
        image, label = mnist[n]
        image = np.asarray(image)
        image = (image >= 128).astype(np.uint8)
        top = tuple(image[:14].flat)
        bottom = tuple(image[14:].flat)
        ddict[top].append(bottom)
    counts = Counter(len(ddict[k]) for k in ddict)
    entropy = 0
    N = len(mnist)
    for K in sorted(counts):
        print(K, counts[K])
        entropy_K = - (K / N) * np.log(K / N)
        entropy += entropy_K * counts[K]
    return entropy

def main():
    # datadir = "/Users/chuck-houyee/Developer/datasets/fashion-mnist"
    datadir = "/Users/chuck-houyee/Developer/datasets/mnist"
    dataset = datasets.MNIST(datadir, train=True)
    entropy = entanglement_entropy(dataset)
    print("Entropy = {}".format(entropy))

if __name__ == '__main__':
    main()

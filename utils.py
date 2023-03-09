import re
import os
import scipy  # for regression estimates
import pandas as pd  # for corr matrix
import numpy as np
from math import ceil
from typing import AnyStr
import random


ASCII_TO_INT: dict = {i.to_bytes(1, 'big'): i for i in range(256)}
INT_TO_ASCII: dict = {i: b for b, i in ASCII_TO_INT.items()}


# open source implementation by https://github.com/BertrandBordage
def LZ(data: AnyStr) -> bytes:
    if isinstance(data, str):
        data = data.encode()
    keys: dict = ASCII_TO_INT.copy()
    n_keys: int = 256
    compressed: list = []
    start: int = 0
    n_data: int = len(data)+1
    while True:
        if n_keys >= 512:
            keys = ASCII_TO_INT.copy()
            n_keys = 256
        for i in range(1, n_data-start):
            w: bytes = data[start:start+i]
            if w not in keys:
                compressed.append(keys[w[:-1]])
                keys[w] = n_keys
                start += i-1
                n_keys += 1
                break
        else:
            compressed.append(keys[w])
            break
    bits: str = ''.join([bin(i)[2:].zfill(9) for i in compressed])
    return int(bits, 2).to_bytes(ceil(len(bits) / 8), 'big')


# does not appear in main
# I used it for preprocessing of the texts
def clean(path):
    string = open(path, encoding="ISO-8859-1").read()
    new_str = re.sub('[^a-zA-Z.,\n ]', '', string.lower())
    os.remove(path)
    open(path, 'w').write(new_str)


# returns list of random fragments of the given text
# k -- starting length of fragments
# n -- max number of fragments
# m -- max number of fragments of the same lenght
def extract_fragments(file_path, k=50, n=250, m=10):
    string = open(file_path).read()
    L = len(string)
    kb = 1024
    cnt = 0
    frags = []
    while len(frags) < n and k * kb < L:
        a = random.randrange(L - k * kb)
        b = a + k * kb
        frags.append(string[a:b])
        cnt += 1
        if not cnt % m:
            k += 2
    return frags


def str_to_len(list_of_str, scale=1):
    return [len(el)/scale for el in list_of_str]


# D^2
def distance(x, y, S):
    return np.dot(np.dot((x - y).T, np.linalg.inv(S)), (x - y)).item(0)


# y = b0 + b1*x + e
# y -- compressed in bytes
# x -- fragment length in kB
def reg_estimates(x, y):
    lr = scipy.stats.linregress(x, y)
    return lr.intercept, lr.slope


def cov_matrix(x, y):
    df = pd.DataFrame({'A': x, 'B': y})
    return pd.DataFrame.cov(df)

# data_loader.py
import torch
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO


class Vocabulary(object):
    """
    Vocabulary class for wrapping the dictionaries that map from word to index and from index to word
    """

    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.idx
            self.idx_to_word[self.idx] = word
            self.idx += 1

    def to_idx(self, word):
        if word not in self.word_to_idx:
            return self.word_to_idx['<<unknown>>']
        return self.word_to_idx[word]

    def to_word(self, idx):
        return self.idx_to_word[idx]

    def __len__(self):
        return len(self.word_to_idx)
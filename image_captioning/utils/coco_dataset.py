import torch
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    '''Dataset class for torch.utils.DataLoader'''
    def __init__(self, image_dir, caption_dir, vocab, transform):
        '''
        Parameters:
        ----------
            image_dir: director to coco image
            caption_dir: coco annotation json file path
            vocab: vocabulary wrapper

        '''
        self.image_dir = image_dir
        # COCO api class that loads COCO annotation file and prepare data structures.
        self.coco = COCO(caption_dir)
        self.keys = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, idx):
        '''
        Private function return one sample, image, caption
        '''
        annotation_id = self.keys[idx]
        image_id = self.coco.anns[annotation_id]['image_id']
        caption = self.coco.anns[annotation_id]['caption']
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']
        # assert img_file_name.split('.')[-1] == 'jpg'

        image = Image.open(os.path.join(self.image_dir, img_file_name)).convert('RGB')
        image = self.transform(image)


        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # append start and end
        caption = [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]
        caption = torch.Tensor(caption)
        return image, caption

    def __len__(self):
        return len(self.keys)

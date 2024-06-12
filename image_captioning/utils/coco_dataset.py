import torch
import torchvision.transforms as transforms
import os
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    """
    Dataset subclass for compatibility with the built-in torch DataLoader
    """

    def __init__(self, image_dir, caption_dir, vocab, transform):
        self.image_dir = image_dir
        self.coco = COCO(caption_dir)
        self.keys = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        annotation_id = self.keys[idx]
        image_id = self.coco.anns[annotation_id]['image_id']
        caption = self.coco.anns[annotation_id]['caption']
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.image_dir, img_file_name)).convert('RGB')
        image = self.transform(image)
        # Convert the caption to a lowercase string and tokenize it into words and punctuation
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # Include the <<start>> and <<end>> tokens for denoting the start and end of the caption
        caption = [self.vocab.to_idx('<<start>>'), *[self.vocab.to_idx(x) for x in tokens], self.vocab.to_idx('<<end>>')]
        caption = torch.Tensor(caption)
        return image, caption

    def __len__(self):
        return len(self.keys)

def coco_batch(coco_data):
    """
    Create mini_batch tensors from the list of tuples, this is to match the output of __getitem__()
    coco_data: list of tuples of length 2:
        coco_data[0]: image, shape of (3, 256, 256)
        coco_data[1]: caption, shape of length of the caption;
    """
    coco_data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*coco_data)

    images = torch.stack(images, 0)

    cap_length = [len(cap) for cap in captions]
    seq_length = max(cap_length)
    if max(cap_length) > 100:
        seq_length = 100
    targets = torch.LongTensor(np.zeros((len(captions), seq_length)))
    for i, cap in enumerate(captions):
        length = cap_length[i]
        targets[i, :length] = cap[:length]

    return images, targets, cap_length
import torch
import torchvision.transforms as transforms
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

#wimport nltk
#from torch.utils.data import Dataset, DataLoader
from PIL import Image
#from generate_vocab_dict import Vocabulary
#from data_loader import CocoDataset, coco_batch

from pycocotools.coco import COCO
import argparse
# from model_V2_dropout0 import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn

from image_captioning import NIC
from image_captioning import Encoder
from image_captioning import Decoder
from image_captioning import Vocabulary
from image_captioning import resize_and_normalize_image


# TO BE DELETED: In generation phase, we need should random crop, just resize


if __name__ == '__main__':

    # Data and model paths and parameters

    width_image_net = 224
    height_image_net = 224
    mean_image_net = [0.485, 0.456, 0.406]
    std_image_net = [0.229, 0.224, 0.225]

    image_path = 'data/ivan.jpg'

    encoder_path = 'checkpoints/encoder.ckpt'
    decoder_path = 'checkpoints/decoder.ckpt'

    vocab_path = 'data/vocab.pkl'

    embed_size = 512
    hidden_size = 512
    num_layers = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    nic = NIC(hidden_size, len(vocab), embed_size, num_layers)
    nic = nic.to(device)
    '''

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(stateful=False, embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    # Load, transform, and move the image to the GPU
    img = Image.open(image_path)
    adjusted_img = resize_and_normalize_image(img, width_image_net, height_image_net, mean_image_net, std_image_net)
    image_tensor = adjusted_img .to(device)


    # Generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
      word = vocab.idx2word[word_id]
      sampled_caption.append(word)
      if word == '<<end>>':
        break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    print(sampled_caption)
    image = Image.open(image_path)

    plt.imshow(np.asarray(image))















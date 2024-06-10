import torch
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

#from torch.utils.data import Dataset, DataLoader
from PIL import Image
#from data_loader import CocoDataset, coco_batch

import argparse


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

    # Create the encoder and decoder instances and set them to evaluation mode
    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers).eval()
    # Load the precomputed parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    # Move the model instances to the device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load and transform the image so it matches the COCO dataset specifications
    img = Image.open(image_path)
    adjusted_img = resize_and_normalize_image(img, width_image_net, height_image_net, mean_image_net, std_image_net)
    # Move the image to the device
    image_tensor = adjusted_img .to(device)

    # Generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.to_word(word_id)
        sampled_caption.append(word)
        if word == '<<end>>':
            break

    sampled_caption = [word for word in sampled_caption if word not in ['<<start>>','<<end>>']]
    sentence = ' '.join(sampled_caption)
    image = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0, sentence, fontsize=12, fontweight='bold')
    plt.imshow(np.asarray(image))
    plt.axis('off')  # Turn off axis for pure text display
    plt.show()
    plt.show()















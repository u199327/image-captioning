import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

from image_captioning import NIC
from image_captioning import Vocabulary
from image_captioning import resize_and_normalize_image
from image_captioning import IMAGENET_IMAGE_SIZE, IMAGENET_IMAGE_MEAN, IMAGENET_IMAGE_STD
from config import NUM_LAYERS, HIDDEN_SIZE, EMBED_SIZE, NIC_PATH, VOCAB_PATH




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Caption generation project.")
    parser.add_argument('--imagepath', type=str, help='Path to the input image')

    args = parser.parse_args()
    image_path = args.imagepath

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    nic = NIC(HIDDEN_SIZE, len(vocab), EMBED_SIZE, NUM_LAYERS).eval()
    nic.load_state_dict(torch.load(NIC_PATH))
    nic = nic.to(device)

    # Load and transform the image, so it matches the COCO dataset specifications
    img = Image.open(image_path)
    adjusted_img = resize_and_normalize_image(img, IMAGENET_IMAGE_SIZE, IMAGENET_IMAGE_SIZE, IMAGENET_IMAGE_MEAN,
                                              IMAGENET_IMAGE_STD)
    # Move the image to the device
    image_tensor = adjusted_img .to(device)

    # Generate the final caption by generating first the corresponding words ids, and the using the Vocabulary to
    # obtain the words
    sampled_ids = nic.generate_caption(image_tensor)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.to_word(word_id)
        sampled_caption.append(word)
        if word == '<<end>>':
            break

    # Show the image and the caption
    sampled_caption = [word for word in sampled_caption if word not in ['<<start>>', '<<end>>']]
    sentence = ' '.join(sampled_caption)
    image = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0, sentence, fontsize=12, fontweight='bold')
    plt.imshow(np.asarray(image))
    plt.axis('off')  # Turn off axis for pure text display
    plt.show()
















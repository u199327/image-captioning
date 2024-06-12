import argparse
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import numpy as np
import logging

from image_captioning import NIC
from image_captioning import Vocabulary
from image_captioning import CocoDataset, coco_batch


def main(args):
    logging.basicConfig(level=logging.INFO)
    log_level = logging.INFO
    logger = logging.getLogger()
    handler = logging.FileHandler("loss_5epochs.log")
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set hyperparameters and paths
    image_dir = args.image_dir
    caption_path = args.caption_path
    vocab_path = args.vocab_path
    model_path = args.model_path
    crop_size = args.crop_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate

    embed_size = 512
    hidden_size = 512
    num_layers = 5
    num_epochs = args.num_epochs
    log_step = 10
    save_step = 300
    max_seq_length = 20

    start_epoch = 0

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    coco = CocoDataset(image_dir, caption_path, vocab, transform)
    dataLoader = DataLoader(coco, batch_size, shuffle=True, num_workers=num_workers, collate_fn=coco_batch)

    nic = NIC(hidden_size, len(vocab), embed_size, num_layers, max_seq_length).to(device)
    nic.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nic.get_parameters(), lr=learning_rate)

    total_step = len(dataLoader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(dataLoader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            outputs = nic(images, captions, lengths)

            nic.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()

            if i % log_step == 0:
                logger.info(('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                             .format(epoch + start_epoch, num_epochs + start_epoch, i, total_step, loss.item(), np.exp(loss.item()))))

            if (i + 1) % save_step == 0:
                torch.save(nic.state_dict(), os.path.join(model_path, 'new_nic-{}-{}.ckpt'.format(epoch + 1 + start_epoch, i + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train image captioning model.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory for images')
    parser.add_argument('--caption_path', type=str, required=True, help='Path for caption file')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path for vocabulary file')
    parser.add_argument('--model_path', type=str, default='model', help='Path for saving models')
    parser.add_argument('--crop_size', type=int, default=224, help='Size for randomly cropping images')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--encoder_path', type=str, help='Path for encoder model')
    parser.add_argument('--decoder_path', type=str, help='Path for decoder model')

    args = parser.parse_args()
    main(args)

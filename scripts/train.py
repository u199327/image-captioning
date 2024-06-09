import argparse
import os
from PIL import Image
import nltk
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import pickle
import numpy as np
import logging

# Ensure nltk punkt tokenizer is downloaded
nltk.download('punkt')

class Encoder(nn.Module):
    """
    Encoder:
        Use the pretrained resnet101 replace the last fc layer and re-train the last fc layer.
    """
    def __init__(self, embed_size=256):
        super(Encoder, self).__init__()

        resnet = models.resnet152(pretrained=True)

        # change the output dimension of the last fc and only requires grad for the weight and bias
        self.resnet = resnet
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        Forward propagation.
        @param
            images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        return:
            image embeddings: (batch_size, embed_size)
        """
        out = self.resnet(images)  # (batch_size, embed_size)
        return out

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.max_seg_length = 20

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs

    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)               # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                 # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)        # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<<unknown>>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class CocoDataset(Dataset):
    '''Dataset class for torch.utils.DataLoader'''
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

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]
        caption = torch.Tensor(caption)
        return image, caption

    def __len__(self):
        return len(self.keys)

    def showImg(sef, idx):
        annotation_id = self.keys[idx]
        image_id = self.coco.anns[annotation_id]['image_id']
        caption = self.coco.anns[annotation_id]['caption']
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']
        assert img_file_name.split('.')[-1] == 'jpg'

        image = Image.open(os.path.join(self.image_dir, img_file_name)).convert('RGB')

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption = [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]

        target = torch.Tensor(caption)
        return image, caption

def coco_batch(coco_data):
    '''
    Create mini_batch tensors from the list of tuples, this is to match the output of __getitem__()
    coco_data: list of tuples of length 2:
        coco_data[0]: image, shape of (3, 256, 256)
        coco_data[1]: caption, shape of length of the caption;
    '''
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
    num_layers = 1
    num_epochs = args.num_epochs
    log_step = 10
    save_step = 30

    encoder_path = args.encoder_path
    decoder_path = args.decoder_path
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

    encoder = Encoder(embed_size=embed_size).to(device)
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers).to(device)

    # Por si queremos usar un modelo pre entrenado
    #encoder.load_state_dict(torch.load(encoder_path))
    #decoder.load_state_dict(torch.load(decoder_path))

    encoder.train()
    decoder.train()
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    total_step = len(dataLoader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(dataLoader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()

            if i % log_step == 0:
                logger.info(('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                             .format(epoch + start_epoch, num_epochs + start_epoch, i, total_step, loss.item(), np.exp(loss.item()))))

            if (i + 1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(model_path, 'new_decoder-{}-{}.ckpt'.format(epoch + 1 + start_epoch, i + 1)))
                torch.save(encoder.state_dict(), os.path.join(model_path, 'new_encoder-{}-{}.ckpt'.format(epoch + 1 + start_epoch, i + 1)))

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

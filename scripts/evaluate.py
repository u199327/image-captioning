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
import pandas as pd
import json

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

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
    def __init__(self, stateful, embed_size, hidden_size, vocab_size, num_layers=1):
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

    def sample(self, features, states=None, beam_width=1):
        """
        Sampling function that generates the beam search approximation for the sequence S that maximizes the probability P(S|I), where I is the input image.
        @param
        features: Latent representation of the image I
        states: Previous hidden and cell states
        beam_width: Beam width used for the beam search
        """
        inputs = features.unsqueeze(1)

        # Beam search initialization. Notice that we manually handle the hidden and cell states
        sequences = [[[], 0.0, inputs, states]]

        for _ in range(self.max_seg_length):
            all_candidates = []
            # Iterate over the candidate sequences
            for seq, score, inputs, states in sequences:
                # Compute the log probabilities for the next word using as input the previous word in the sequence
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1))
                log_probs = torch.log_softmax(outputs, dim=1)

                # Get top beam_width candidate words
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=1)

                # Add the top beam_width candidate words to the candidate sequence and save the information in the all_candidates array
                for i in range(beam_width):
                    candidate = [seq + [top_indices[0][i].item()],
                                (score + top_log_probs[0][i].item()) / len(seq + [top_indices[0][i].item()]),
                                self.embed(top_indices[:, i]).unsqueeze(1),
                                states]
                    all_candidates.append(candidate)

            # Order all candidates by score and select the top beam_width sequences
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]

        # Select the sequence with the highest score
        best_sequence = max(sequences, key=lambda x: x[1])[0]
        best_sequence = torch.tensor(best_sequence)
        return best_sequence.unsqueeze(0)

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


# get imagename dataframe reference, jsonPath is the validation set json file path
def get_image_name(jsonPath):
    # python 3
    with open(jsonPath, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    dataImage = pd.DataFrame.from_dict(data['images'])
    dataAnnotations = pd.DataFrame.from_dict(data['annotations'])
    dataName = pd.merge(dataImage,dataAnnotations, left_on='id',right_on='image_id')
    dataName = dataName[['file_name','caption']]
    return dataName

# name_caption_frame = get_image_name(jsonPath)

# define bleu score calculation, imgpath should be like /example/example.jpg, captions should be like ['<start>','example','example','<end>']
def bleu_score(input_imgs_path, generated_captions, name_caption_frame):
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
#    print(captions)
    references = []
    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)
    candidates = generated_captions[1:-1]
    generated_score = sentence_bleu(references, candidates, smoothing_function=SmoothingFunction().method4)
    theoratical_score = 0
    for i in range(5):
        theoratical_score += sentence_bleu(references[:i]+references[i+1:], references[i], smoothing_function=SmoothingFunction().method4)
        #print(references[:i]+references[i+1:],theoratical_score)
    theoratical_score /= 5.0
    return generated_score,theoratical_score


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def generate_caption(image_path, vocab, encoder, decoder, transform):
    # Image preprocessing
    image = load_image(image_path, transform)
    image_tensor = image.to('cuda')

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<<end>>':
            break

    return sampled_caption

def test(args):
    # Define paths
    jsonPath = './filtered_captions_train2014_v2.json'
    image_dir = './resized_val2014'


    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)

    # Build model
    encoder = Encoder(embed_size=args['embed_size']).eval()
    decoder = Decoder(stateful=False, embed_size=args['embed_size'], hidden_size=args['hidden_size'], vocab_size=len(vocab), num_layers=args['num_layers']).eval()
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args['encoder_path'], map_location='cuda'))
    decoder.load_state_dict(torch.load(args['decoder_path'], map_location='cuda'))

    name_caption_frame = get_image_name(jsonPath)
    unique_image_names = pd.unique(name_caption_frame['file_name'])

    # Add image directory path train2014 or val2014
    unique_image_names = [os.path.join(image_dir, image_name) for image_name in unique_image_names]

    total_generated_score = []
    total_theoretical_score = []

    # Parallelize the process
    def score_helper(image_path):
        caption = generate_caption(image_path, vocab, encoder, decoder, transform)
        generated_score, theoretical_score = bleu_score(image_path, caption, name_caption_frame)
        total_generated_score.append(generated_score)
        total_theoretical_score.append(theoretical_score)
        #print(generated_score, theoretical_score)

    _ = pd.Series(unique_image_names).apply(score_helper)

    print('Average bleu score:', sum(total_generated_score) / len(total_generated_score),
          ' | Average theoretical score:', sum(total_theoretical_score) / len(total_theoretical_score))

    return total_generated_score, total_theoretical_score

if __name__ == "__main__":
    # Define arguments
    args = {
        'eval': 'eval',  # or 'train'
        'encoder_path': './new_encoder-5-4000.ckpt',
        'decoder_path': './new_decoder-5-4000.ckpt',
        'vocab_path': './vocab.pkl',
        'embed_size': 512,
        'hidden_size': 512,
        'num_layers': 1
    }

    print(args['encoder_path'])

    # Run the test function
    total_generated_score, total_theoretical_score = test(args)
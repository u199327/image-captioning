import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class NIC(nn.Module):
    """
    Implementation of the Neural Image Caption (NIC) defined in the paper "Show and Tell: A Neural Image Caption Generator"
    Wrapper of the encoder and decoder classes.
    """

    def __init__(self, hidden_size, vocab_size, embed_size=256,  num_layers=1, max_seq_length=20):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)

    def forward(self, images, captions, lengths):
        image_features = self.encoder(images)
        caption_ids = self.decoder(image_features, captions, lengths)
        return caption_ids

    def generate_caption(self, images, states=None, beam_width=1):
        features = self.encoder(images)
        sampled_ids = self.decoder.sample(features, states, beam_width)
        return sampled_ids

    def get_parameters(self):
        return list(self.decoder.parameters()) + list(self.encoder.resnet.fc.parameters())

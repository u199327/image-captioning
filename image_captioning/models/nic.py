import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

'''
Implementation of the Neural Image Caption (NIC) defined in the paper "Show and Tell: A Neural Image Caption Generator"
'''


class NIC(nn.Module):
    def __init__(self, hidden_size, vocab_size, attention=False, embed_size=256, encoded_image_size=14,  num_layers=1, max_seq_length=20, stateful=True):
        super().__init__()
        self.encoder = Encoder(attention, embed_size, encoded_image_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, max_seq_length, stateful)

    def forward(self, images):
        image_features = self.encoder(images)
        caption_ids = self.decoder(image_features)
        return caption_ids

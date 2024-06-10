import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Encoder:
        Use the pretrained ResNet152 Convolutional Neural Network and retrain the last fully connected layer to obtain the
        feature representation of the images.
    """
    # TO BE DELETED: attention, encoded_image_size
    def __init__(self, embed_size=256):
        super().__init__()
        # Load the pretrained CNN ResNet152
        self.resnet = models.resnet152(weights='DEFAULT')
        # Fix the weights of the CNN, so they are not updated during the training
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Reinitialize the last fully connected layer of the CNN, and set the output dimension to the size of the words embedding
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        Forward propagation.
        @param
            images: Tensor of dimensions (batch_size, 3, image_size, image_size) representing a batch of RGB images
        return:
            image embeddings: Tensor of dimensions (batch_size, embed_size) representing the feature vectors of the input images
        """
        out = self.resnet(images)
        return out

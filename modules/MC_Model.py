# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Channel model
class MC_Model(nn.Module):
    """
    Multi-channel with pre-trained layers
    """

    # Constructor
    def __init__(self, text_layer, image_layer, n_texts=100, n_images=10):
        """
        Constructor
        :param text_layer: Pre-trained text layer
        :param image_layer: Pre-trained image layer
        :param n_texts: Number of text in the inputs
        :param n_images: Number of images in the inputs
        """
        # Properties
        self.text_layer = text_layer
        self.image_layer = image_layer
        self.n_texts = n_texts
        self.n_images = n_images

        # Number of features of the final fully connected layer of the text pre-trained network
        text_num_ftrs = text_layer.fc.in_features

        # Number of features of the final fully connected layer of the image
        image_num_ftrs = image_layer.fc.in_features

        # Linear layer
        self.linear_size = text_num_ftrs * n_texts + image_num_ftrs * n_images
        self.linear = nn.Linear(self.linear_size, 2)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x: Data
        :return:
        """
        pass
    # end forward

# end CNNMC

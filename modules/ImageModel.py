# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import echotorch.nn as etnn


# Image model
class ImageModel(nn.Module):
    """
    Image model
    """

    # Constructor
    def __init__(self, image_model, n_images=10):
        """
        Constructor
        :param image_model: Single image model
        :param n_images: Number of images
        """
        # Super
        super(ImageModel, self).__init__()

        # Properties
        self.image_model = image_model
        self.n_images = n_images

        # Number of features of the final fully connected layer of the image
        image_num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = etnn.Identity()

        # Linear layer
        self.linear_size = image_num_ftrs * n_images
        # self.linear1 = nn.Linear(self.linear_size, image_num_ftrs)
        # self.linear2 = nn.Linear(image_num_ftrs, 2)
        self.linear = nn.Linear(self.linear_size, 2)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x: Data
        :return:
        """
        # Make a whole batch
        x = x.view(-1, 3, 224, 224)

        # Apply image model
        x = self.image_model(x)

        # Remake batch
        x = x.view(-1, 5120)

        # Apply two linear layers
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = F.relu(self.linear(x))

        # Return log softmax
        return F.log_softmax(x, dim=1)
    # end forward

# end CNNMC

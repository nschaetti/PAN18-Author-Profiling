# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchvision import transforms
import dataset
from echotorch.transforms import text


# Settings
batch_size = 1
image_size = 300

# Text tranformer
text_transform = text.Character()

# Image transformer
image_transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

# Author profiling training data sets
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorProfilingDataset(min_length=165, root="./data/", download=True, text_transform=text_transform,
                                   image_transform=image_transform, image_size=image_size),
    batch_size=batch_size, shuffle=True)

# For each samples
for data in pan18loader_training:
    # Inputs and labels
    tweets, images, label = data
    """print(tweets.size())
    print(images.size())
    print(label.size())"""
# end for

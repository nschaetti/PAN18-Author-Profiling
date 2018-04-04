# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchvision import transforms
import dataset
from echotorch.transforms import text


# Settings
batch_size = 64
image_size = 300

# Text tranformer
text_transform = text.Character()

# Tweet data set
tweet_dataset = dataset.TweetDataset(min_length=165, root='./data/', download=True, text_transform=text_transform)

# Author profiling training data sets
pan17loader_training = torch.utils.data.DataLoader(tweet_dataset, batch_size=batch_size, shuffle=True, year=2017)

# For each samples
for data in pan18loader_training:
    # Inputs and labels
    inputs, labels = data
    print(inputs.size())
    print(labels.size())
# end for

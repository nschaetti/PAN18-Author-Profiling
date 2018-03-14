# -*- coding: utf-8 -*-
#

# Imports
import torch
import dataset
from echotorch.transforms import text


# Settings
batch_size = 64

# Author profiling training data sets
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorProfilingDataset(root="./data/", download=True),
    batch_size=batch_size, shuffle=True)

# For each samples
"""for data in pan18loader_training:
    # Inputs and labels
    inputs, images, labels = data
    print(inputs.size())
    print(labels.size())
# end for"""

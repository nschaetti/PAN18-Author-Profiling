# -*- coding: utf-8 -*-
#

# Imports
import torch
import dataset
from echotorch.transforms import text


# Settings
batch_size = 1

# Author profiling training data sets
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorProfilingDataset(root="./data/", download=True),
    batch_size=batch_size, shuffle=True)

# For each samples
for data in pan18loader_training:
    # Inputs and labels
    tweets, images, label = data
    print(tweets)
    print(images)
    print(label)
    print(u"")
# end for

# -*- coding: utf-8 -*-
#

# Imports
import torch
import dataset
from echotorch.transforms import text


# Settings
batch_size = 1

# Text tranformer
text_transform = text.Character()

# Author profiling training data sets
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorProfilingDataset(min_length=155, root="./data/", download=True, text_transform=text_transform),
    batch_size=batch_size, shuffle=True)

# For each samples
for data in pan18loader_training:
    # Inputs and labels
    tweets, label = data
    print(tweets)
    print(label)
    print(u"")
# end for

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

# Image transformer
image_transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

# Image data set
image_dataset = dataset.ImageDataset(root='./data', download=True, image_transform=image_transform, image_size=image_size)

# Tweet data set
tweet_dataset = dataset.TweetDataset(min_length=165, root='./data/', download=True, text_transform=text_transform)

# Author profiling training data sets
pan18loader_training = torch.utils.data.DataLoader(tweet_dataset, batch_size=batch_size, shuffle=True)

# For each samples
for data in image_dataset:
    # Inputs and labels
    images, label = data
    print(tweets.size())
    print(images.size())
    print(label.size())
# end for

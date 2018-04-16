# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchlanguage import transforms as ltransforms
import dataset
import argparse
from torchvision import transforms
from torch.autograd import Variable
import os
from tools import settings, functions
import numpy as np


################################################
# MAIN
################################################

# Argument parser
args = functions.argument_parser_execution()

# Load models and voc
image_model, tweet_model, tweet_voc = functions.load_models(args.image_model, args.tweet_model_dir, 'en', args.cuda)

# Image augmentation and normalization
image_transform = functions.image_transformer('val')

# Transformer
text_transform = functions.tweet_transformer(args.lang, args.n_gram)

# Author profiling data set
profiling_dataset = dataset.AuthorProfilingDataset(root='./data/', download=True, lang=args.lang,
                                                   text_transform=text_transform, image_transform=image_transform,
                                                   train=True, val=0)
pan18loader = torch.utils.data.DataLoader(profiling_dataset, batch_size=args.batch_size, shuffle=True)

# Counters
images_success = 0.0
tweets_success = 0.0
both_success = 0.0
total = 0.0

# For validation set
count = 0
for data in pan18loader:
    # Images, tweets and labels
    tweets, images, labels = data

    # Variable and CUDA
    images, tweets, labels = Variable(images), Variable(tweets), Variable(labels)
    if args.cuda:
        images, tweets, labels = images.cuda(), tweets.cuda(), labels.cuda()
    # end if

    # Remove images and tweet dimension
    tweets = tweets.view(-1, settings.min_length)
    images = images.view(-1, 3, settings.image_size, settings.image_size)

    # Compute prob. for tweets and images
    images_probs = image_model(images)
    tweets_probs = tweet_model(tweets)

    # Resize to images and tweet dimension and transpose
    tweets_probs = tweets_probs.view(-1, 100, 2)
    images_probs = images_probs.view(-1, 10, 2)

    # Take the max as predicted
    _, images_prediction = torch.max(torch.mean(images_probs, 1), 1)
    _, tweets_prediction = torch.max(torch.mean(tweets_probs, 1), 1)

    # Both prediction
    _, both_prediction = torch.max((torch.mean(images_probs, 1) * settings.alpha + torch.mean(tweets_probs, 1) * (1.0 - settings.alpha)) / 2.0, 1)

    # Add to correctly classified profiles
    images_success += (images_prediction == labels).sum()
    tweets_success += (tweets_prediction == labels).sum()
    both_success += (both_prediction == labels).sum()

    # Add to total
    total += labels.size(0)
    count += labels.size(0)

    # Save result
    for i in range(args.batch_size):
        author_id = profiling_dataset.last_idxs[-args.batch_size+i]
        functions.save_result(
            args.output,
            author_id,
            args.lang,
            tweets_prediction[i],
            images_prediction[i],
            both_prediction[i]
        )
    # end for
# end for

# Accuracies
images_accuracy = images_success / total * 100.0
tweets_accuracy = tweets_success / total * 100.0
both_accuracy = both_success / total * 100.0

# Print and save loss
print(u"Images accuracy {}, Tweets accuracy {}, Both accuracy {}".format(images_accuracy, tweets_accuracy, both_accuracy))

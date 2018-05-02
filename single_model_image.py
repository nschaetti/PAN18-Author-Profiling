#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : single_image_tweet.py
# Description : Train pre-trained model on images.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Neuchâtel, Suisse
#
# This file is part of the PAN18 author profiling challenge code.
# The PAN18 author profiling challenge code is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import torch
from torchvision import models, transforms
import dataset
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
from tools import functions, settings


def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
# end imsho


# Argument parser
args = functions.argument_parser_training_model('image')

# Image augmentation and normalization
image_transforms = dict()
image_transforms['train'] = functions.image_transformer('train')
image_transforms['val'] = functions.image_transformer('val')

# Image data set
pan18loader_training, pan18loader_validation = functions.load_images_dataset(
    image_transforms,
    args.batch_size, args.val_batch_size
)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Model
if args.model == 'resnet18':
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
else:
    model = models.alexnet(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 2),
    )
# end if
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Epoch
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0
    test_loss = 0.0
    test_total = 0.0
    image_count = 0

    # For each training samples
    for data in pan18loader_training:
        # Inputs and labels
        images, labels = data

        # Variable and CUDA
        images, labels = Variable(images), Variable(labels)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        # end if

        # Zero grad
        model.zero_grad()

        # Compute output
        log_probs = model(images)

        # Loss
        loss = loss_function(log_probs, labels)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Add
        training_loss += loss.data[0]
        training_total += 1.0
        image_count += images.size(0)
        if args.training_count != -1 and image_count >= args.training_image_count:
            break
        # end if
    # end for

    # Counters
    total = 0.0
    success = 0.0

    # For each validation samples
    image_count = 0
    for data in pan18loader_validation:
        # Inputs and labels
        images, labels = data

        # Variable and CUDA
        images, labels = Variable(images), Variable(labels)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        # end if

        # Forward
        model_outputs = model(images)

        # Compute loss
        loss = loss_function(model_outputs, labels)

        # Take the max as predicted
        _, predicted = torch.max(model_outputs.data, 1)

        # Add to correctly classified word
        success += (predicted == labels.data).sum()
        total += predicted.size(0)

        # Add loss
        test_loss += loss.data[0]
        test_total += 1.0
        image_count += images.size(0)
        if args.training_count != -1 and image_count >= args.test_image_count:
            break
        # end if
    # end for

    # Accuracy
    accuracy = success / total * 100.0

    # Print and save loss
    print(u"Epoch {}, training loss {}, test loss {}, accuracy {}".format(epoch, training_loss / training_total,
                                                                          test_loss / test_total, accuracy))

    # Save if better
    if accuracy > best_acc:
        best_acc = accuracy
        print(u"Saving model with best accuracy {}".format(best_acc))
        torch.save(model.state_dict(), open(args.output, 'wb'))
    # end if
# end for

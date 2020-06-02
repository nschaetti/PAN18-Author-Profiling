#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : single_model_tweet.py
# Description : Train CNN on Tweet.
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
from torchlanguage import models
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import copy
import os
from tools import functions, settings

# Parse argument
args = functions.argument_parser_training_model('tweet')

# Transformer
transformer = functions.tweet_transformer(args.lang, args.n_gram)

# Load data sets
pan17loader_training, pan17loader_validation, pan18loader_training, pan18loader_validation = \
    functions.load_tweets_dataset(args.lang, transformer, args.batch_size, args.val_batch_size)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Model
model = models.CNNCTweet(text_length=settings.min_length, vocab_size=settings.voc_sizes[args.n_gram][args.lang],
                         embedding_dim=args.dim)
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

    # Training
    model.train()

    # For each training set
    # for training_set in [pan17loader_training, pan17loader_validation, pan18loader_training]:
    for training_set in [pan18loader_training]:
        count = 0
        for data in training_set:
            # Inputs and labels
            inputs, labels = data

            # Batch size
            data_batch_size = inputs.size(0)

            # Merge batch and authors
            inputs = inputs.view(-1, settings.min_length)
            labels = labels.view(data_batch_size * 100)

            # Variable and CUDA
            inputs, labels = Variable(inputs), Variable(labels)
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # end if

            # Zero grad
            model.zero_grad()

            # Compute output
            log_probs = model(inputs)

            # Loss
            loss = loss_function(log_probs, labels)

            # Backward and step
            loss.backward()
            optimizer.step()

            # Add
            training_loss += loss.data[0]
            training_total += 1.0
            count += inputs.size(0)
            if args.training_count != -1 and count >= int(args.training_count / 3):
                break
            # end if
        # end for
    # end for

    # Counters
    total = 0.0
    success = 0.0

    # Eval
    model.eval()

    # For validation set
    count = 0
    for data in pan18loader_validation:
        # Inputs and labels
        inputs, labels = data

        # Merge batch and authors
        inputs = inputs.view(-1, settings.min_length)
        labels = labels.view(args.val_batch_size * 100)

        # Variable and CUDA
        inputs, labels = Variable(inputs), Variable(labels)
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # end if

        # Forward
        model_outputs = model(inputs)

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
        count += inputs.size(0)
        if args.test_count != -1 and count >= args.test_count:
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
        torch.save(
            transformer.transforms[3].token_to_ix,
            open(os.path.join(args.output, "voc_" + args.lang + ".p"), 'wb')
        )
        torch.save(
            model.state_dict(),
            open(os.path.join(args.output, args.lang + ".p"), 'wb')
        )
    # end if
# end for

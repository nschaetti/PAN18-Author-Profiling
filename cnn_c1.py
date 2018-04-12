# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchlanguage import transforms
from torchlanguage import models
import dataset
import argparse
import torch.nn as nn
from modules import CNNT
from torch.autograd import Variable
from torch import optim
import copy
import os


# Settings
batch_size = 16
min_length = 165
voc_size = 1580

# Argument parser
parser = argparse.ArgumentParser(description="PAN18 Author Profiling CNN-C1")

# Argument
parser.add_argument("--output", type=str, help="Model output file", default='.')
parser.add_argument("--dim", type=int, help="Embedding dimension", default=30)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--lang", type=str, help="Language", default='en')
parser.add_argument("--training-tweet-count", type=int, help="Number of tweets to train", default=2000)
parser.add_argument("--test-tweet-count", type=int, help="Number of tweets to test", default=200)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Text tranformer
text_transform = transforms.Compose([
    transforms.RemoveRegex(regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    transforms.ToLower(),
    transforms.Character(),
    transforms.ToIndex(start_ix=1)
])

# Tweet data set 2017 training
tweet_dataset_train_17 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True, lang=args.lang,
                                              text_transform=text_transform, year=2017, train=True)
pan17loader_training = torch.utils.data.DataLoader(tweet_dataset_train_17, batch_size=batch_size, shuffle=True)

# Tweet data set 2017 validation
tweet_dataset_val_17 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True, lang=args.lang,
                                            text_transform=text_transform, year=2017, train=False)
pan17loader_validation = torch.utils.data.DataLoader(tweet_dataset_val_17, batch_size=batch_size, shuffle=True)

# Tweet data set 2018 training
tweet_dataset_train_18 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True, lang=args.lang,
                                              text_transform=text_transform, year=2018, train=True)
pan18loader_training = torch.utils.data.DataLoader(tweet_dataset_train_18, batch_size=batch_size, shuffle=True)

# Tweet data set 2018 validation
tweet_dataset_val_18 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True, lang=args.lang,
                                            text_transform=text_transform, year=2018, train=False)
pan18loader_validation = torch.utils.data.DataLoader(tweet_dataset_val_18, batch_size=batch_size, shuffle=True)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Model
# model = CNNT(vocab_size=voc_size, embedding_dim=args.dim)
model = models.CNNCTweet(text_length=min_length, vocab_size=voc_size, embedding_dim=args.dim)
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

    # For each training set
    for training_set in [pan17loader_training, pan17loader_validation, pan18loader_training]:
        count = 0
        for data in training_set:
            # Inputs and labels
            inputs, labels = data

            # Batch size
            data_batch_size = inputs.size(0)

            # Merge batch and authors
            inputs = inputs.view(-1, min_length)
            labels = labels.view(data_batch_size * 100)

            # Variable and CUDA
            inputs, labels = Variable(inputs), Variable(labels)
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # end if

            # Zero grad
            model.zero_grad()

            # Compute output
            try:
                log_probs = model(inputs)
            except RuntimeError:
                print(inputs.size())
                exit()
            # end try

            # Loss
            loss = loss_function(log_probs, labels)

            # Backward and step
            loss.backward()
            optimizer.step()

            # Add
            training_loss += loss.data[0]
            training_total += 1.0
            count += inputs.size(0)
            if count >= int(args.training_tweet_count / 3):
                break
            # end if
        # end for
        print(training_total)
    # end for

    # Counters
    total = 0.0
    success = 0.0

    # For validation set
    count = 0
    for data in pan18loader_validation:
        # Inputs and labels
        inputs, labels = data

        # Merge batch and authors
        inputs = inputs.view(-1, min_length)
        labels = labels.view(batch_size * 100)

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
        if count >= args.test_tweet_count:
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
        best_model = copy.deepcopy(model.state_dict())
    # end if
# end for

# Load best model
model.load_state_dict(best_model)

# Save
torch.save(text_transform.transforms[2].token_to_ix, open(os.path.join(args.output, "voc.p"), 'wb'))
torch.save(model, open(os.path.join(args.output, "model.p"), 'wb'))

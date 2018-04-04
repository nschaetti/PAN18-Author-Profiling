# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchvision import transforms
import dataset
from echotorch.transforms import text
import argparse
import torch.nn as nn
from modules import CNNT
from torch.autograd import Variable
from torch import optim
import copy


# Settings
batch_size = 5
image_size = 100
min_length = 165
n_epoch = 1
voc_size = 1000

# Argument parser
parser = argparse.ArgumentParser(description="PAN18 Author Profiling CNN-C1")

# Argument
parser.add_argument("--output", type=str, help="Model output file", default='.')
parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Text tranformer
text_transform = text.Character2Gram()

# Tweet data set 2017 training
tweet_dataset_train_17 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True,
                                              text_transform=text_transform, year=2017, train=True)
pan17loader_training = torch.utils.data.DataLoader(tweet_dataset_train_17, batch_size=batch_size, shuffle=True)

# Tweet data set 2017 validation
tweet_dataset_val_17 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True,
                                            text_transform=text_transform, year=2017, train=False)
pan17loader_validation = torch.utils.data.DataLoader(tweet_dataset_val_17, batch_size=batch_size, shuffle=True)

# Tweet data set 2018 training
tweet_dataset_train_18 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True,
                                              text_transform=text_transform, year=2018, train=True)
pan18loader_training = torch.utils.data.DataLoader(tweet_dataset_train_18, batch_size=batch_size, shuffle=True)

# Tweet data set 2018 validation
tweet_dataset_val_18 = dataset.TweetDataset(min_length=min_length, root='./data/', download=True,
                                            text_transform=text_transform, year=2018, train=False)
pan18loader_validation = torch.utils.data.DataLoader(tweet_dataset_val_18, batch_size=batch_size, shuffle=True)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Model
model = CNNT(vocab_size=voc_size, embedding_dim=args.dim)
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Epoch
for epoch in range(n_epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0
    test_loss = 0.0
    test_total = 0.0

    # For each training set
    for training_set in [pan17loader_training, pan17loader_validation, pan18loader_training]:
        for data in training_set:
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
        # end for
    # end for

    # Counters
    total = 0.0
    success = 0.0

    # For validation set
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
torch.save((text_transform.gram_to_ix, model), open(args.output, 'wb'))

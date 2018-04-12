# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchlanguage import transforms as ltransforms
from torchlanguage import models
import dataset
import argparse
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import optim
import copy
import os


# Settings
batch_size = 16
image_size = 224
min_length = 165

# Argument parser
parser = argparse.ArgumentParser(description="PAN18 Author Profiling join model")

# Argument
parser.add_argument("--output", type=str, help="Model output file", required=True)
parser.add_argument("--image-model", type=str, help="Image model", required=True)
parser.add_argument("--tweet-model", type=str, help="Image model", required=True)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--lang", type=str, help="Language", default='en')
parser.add_argument("--training-tweet-count", type=int, help="Number of tweets to train", default=2000)
parser.add_argument("--test-tweet-count", type=int, help="Number of tweets to test", default=200)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Text tranformer
text_transform = ltransforms.Compose([
    ltransforms.RemoveRegex(regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    ltransforms.ToLower(),
    ltransforms.Character(),
    ltransforms.ToIndex(start_ix=1)
])

# Image augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        # transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ]),
}

# Author profiling data set
profiling_dataset_train_18 = dataset.AuthorProfilingDataset(min_length=min_length, root='./data/', download=True,
                                                            lang=args.lang, text_transform=text_transform,
                                                            image_transform=data_transforms, train=True)
pan18loader_training = torch.utils.data.DataLoader(profiling_dataset_train_18, batch_size=batch_size, shuffle=True)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Model
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
    test_total = 0.

    # For each training set
    for training_set in pan18loader_training:
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

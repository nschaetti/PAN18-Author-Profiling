# -*- coding: utf-8 -*-
#

# Imports
import torch
from torchvision import transforms
from torchlanguage import models
from torchlanguage import transforms as ltransforms
import dataset
import argparse
import torch.nn as nn
from modules import ImageModel
from torch.autograd import Variable
from torch import optim
import copy
import os


# Settings
image_size = 224
min_length = 165

# Argument parser
parser = argparse.ArgumentParser(description="PAN18 Author Profiling CNN-C1")

# Argument
parser.add_argument("--image-model", type=str, help="Image model file", required=True)
parser.add_argument("--output", type=str, help="Model output file", required=True)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--batch-size", type=int, help="Batch size", default=20)
parser.add_argument("--val-batch-size", type=int, help="Val. batch size", default=5)
parser.add_argument("--training-image-count", type=int, help="Number of images to train", default=-1)
parser.add_argument("--test-image-count", type=int, help="Number of images to test", default=-1)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Text tranformer
text_transform = transforms.Compose([
    ltransforms.RemoveRegex(regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    ltransforms.ToLower(),
    ltransforms.Character(),
    ltransforms.ToIndex(start_ix=1)
])

# Image augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# PAN 18 profiling training data set
profiling_dataset_train = dataset.AuthorProfilingDataset(min_length=min_length, root='./data/', download=True,
                                                         lang='en', text_transform=text_transform,
                                                         image_transform=data_transforms['train'], train=True)
pan18loader_training = torch.utils.data.DataLoader(profiling_dataset_train, batch_size=args.batch_size, shuffle=True)

# PAN 18 profiling test data set
profiling_dataset_val = dataset.AuthorProfilingDataset(min_length=min_length, root='./data/', download=True,
                                                       lang='en', text_transform=text_transform,
                                                       image_transform=data_transforms['val'], train=False)
pan18loader_validation = torch.utils.data.DataLoader(profiling_dataset_val, batch_size=args.val_batch_size, shuffle=True)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Load image model
image_model = torch.load(open(args.image_model, 'rb'))

# No parameters
for param in image_model.parameters():
    param.requires_grad = False
# end for

# Model
model = ImageModel(image_model=image_model, n_images=10)

# Make CUDA
if args.cuda:
    model.cuda()
# end if

best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.linear.parameters(), lr=0.001, momentum=0.9)

# Epoch
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0
    test_loss = 0.0
    test_total = 0.0
    count = 0

    # For each training set
    for data in pan18loader_training:
        # Inputs and labels
        tweets, images, labels = data

        # Variable and CUDA
        images, labels = Variable(images), Variable(labels)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        # end if

        # Zero grad
        model.zero_grad()

        # Call model
        log_probs = model(images)

        # Loss
        loss = loss_function(log_probs, labels)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Add
        training_loss += loss.data[0]
        training_total += 1.0

        # Check count
        count += images.size(0)
        if args.training_image_count != -1 and count >= args.training_image_count:
            break
        # end if
    # end for

    # Counters
    total = 0.0
    success = 0.0

    # For validation set
    count = 0
    for data in pan18loader_validation:
        # Inputs and labels
        tweets, images, labels = data

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

        # Check count
        count += images.size(0)
        if args.test_image_count != -1 and count >= args.test_image_count:
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
torch.save(model, open(args.output, 'wb'))

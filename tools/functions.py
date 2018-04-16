# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchvision import transforms
import argparse
import dataset
import torch
import settings
import os
import codecs

#################
# Arguments
#################


# Tweet argument parser for training model
def argument_parser_training_model(model_type='tweet'):
    """
    Tweet argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Author Profiling challenge")

    # Argument
    parser.add_argument("--output", type=str, help="Model output file", default='.')
    if model_type == 'tweet':
        parser.add_argument("--dim", type=int, help="Embedding dimension", default=30)
    # end if
    if model_type == 'tweet':
        parser.add_argument("--n-gram", type=str, help="N-Gram (c1, c2)", default='c1')
    # end if
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    parser.add_argument("--epoch", type=int, help="Epoch", default=300)
    if model_type == 'tweet':
        parser.add_argument("--lang", type=str, help="Language", default='en')
    # end if
    parser.add_argument("--batch-size", type=int, help="Batch size", default=20)
    parser.add_argument("--val-batch-size", type=int, help="Val. batch size", default=5)
    parser.add_argument("--training-count", type=int, help="Number of samples to train", default=-1)
    parser.add_argument("--test-count", type=int, help="Number of samples to test", default=-1)
    args = parser.parse_args()

    # Use CUDA?
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
# end argument_parser_training_model


# Execution argument parser
def argument_parser_execution():
    """
    Execution argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Author Profiling main program")

    # Argument
    parser.add_argument("--output", type=str, help="Where to put results", required=True)
    parser.add_argument("--image-model", type=str, help="Image model", required=True)
    parser.add_argument("--tweet-model-dir", type=str, help="Tweet model directory", required=True)
    parser.add_argument("--n-gram", type=str, help="N-Gram (c1, c2)", default='c1')
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    parser.add_argument("--lang", type=str, help="Language", default='en')
    parser.add_argument("--batch-size", type=int, help="Batch size", default=1)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
# end argument_parser_execution

#################
# Transformers
#################


# Get tweet transformer
def tweet_transformer(lang, n_gram):
    """
    Get tweet transformer
    :param lang:
    :param n_gram:
    :return:
    """
    if n_gram == 'c1':
        return transforms.Compose([
            ltransforms.RemoveRegex(
                regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            ltransforms.ToLower(),
            ltransforms.Character(),
            ltransforms.ToIndex(start_ix=1),
            ltransforms.ToLength(length=settings.min_length),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram][lang] - 1)
        ])
    else:
        return transforms.Compose([
            ltransforms.RemoveRegex(
                regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            ltransforms.ToLower(),
            ltransforms.Character2Gram(),
            ltransforms.ToIndex(start_ix=1),
            ltransforms.ToLength(length=settings.min_length),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram][lang] - 1)
        ])
    # end if
# end tweet_transformer


# Get image transformer
def image_transformer(train='train'):
    """
    # Get image transformer
    :param train:
    :return:
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(settings.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(settings.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(settings.image_size),
            transforms.CenterCrop(settings.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[train]
# end image_transformer

#################
# Dataset
#################


# Import tweets data set
def load_tweets_dataset(lang, text_transform, batch_size, val_batch_size):
    """
    Import tweets data set
    :param lang:
    :param text_transform:
    :param batch_size:
    :param val_batch_size:
    :return:
    """
    # Tweet data set 2017 training
    tweet_dataset_train_17 = dataset.TweetDataset(root='./data/', download=True, lang=lang,
                                                  text_transform=text_transform, year=2017, train=True)
    pan17loader_training = torch.utils.data.DataLoader(tweet_dataset_train_17, batch_size=batch_size, shuffle=True)

    # Tweet data set 2017 validation
    tweet_dataset_val_17 = dataset.TweetDataset(root='./data/', download=True, lang=lang,
                                                text_transform=text_transform, year=2017, train=False)
    pan17loader_validation = torch.utils.data.DataLoader(tweet_dataset_val_17, batch_size=batch_size, shuffle=True)

    # Tweet data set 2018 training
    tweet_dataset_train_18 = dataset.TweetDataset(root='./data/', download=True,
                                                  lang=lang, text_transform=text_transform, year=2018,
                                                  train=True)
    pan18loader_training = torch.utils.data.DataLoader(tweet_dataset_train_18, batch_size=batch_size, shuffle=True)

    # Tweet data set 2018 validation
    tweet_dataset_val_18 = dataset.TweetDataset(root='./data/', download=True,
                                                lang=lang, text_transform=text_transform, year=2018,
                                                train=False)
    pan18loader_validation = torch.utils.data.DataLoader(tweet_dataset_val_18, batch_size=val_batch_size,
                                                         shuffle=True)

    return pan17loader_training, pan17loader_validation, pan18loader_training, pan18loader_validation
# end load_tweet_dataset


# Import images data set
def load_images_dataset(image_transforms, batch_size, val_batch_size):
    """

    :return:
    """
    # Image data set training
    image_dataset_training = dataset.ImageDataset(root='./data', download=True,
                                                  image_transform=image_transforms['train'],
                                                  image_size=settings.image_size, train=True)
    pan18loader_training = torch.utils.data.DataLoader(image_dataset_training, batch_size=batch_size, shuffle=True)

    # Image data set validation
    image_dataset_validation = dataset.ImageDataset(root='./data', download=True,
                                                    image_transform=image_transforms['val'],
                                                    image_size=settings.image_size, train=False)
    pan18loader_validation = torch.utils.data.DataLoader(image_dataset_validation, batch_size=val_batch_size, shuffle=True)
    return pan18loader_training, pan18loader_validation
# end load_images_dataset

################
# Models
################


# Load models
def load_models(image_model_file, tweet_model_dir, lang, cuda=False):
    """
    Load models
    :param image_model_file:
    :param tweet_model_dir:
    :param cuda:
    :return:
    """
    # Load image model
    image_model = torch.load(open(image_model_file, 'rb'))
    if cuda:
        image_model.cuda()
    else:
        image_model.cpu()
    # end if

    # Load tweet model
    tweet_model = torch.load(open(os.path.join(tweet_model_dir, lang + ".p"), 'rb'))
    if cuda:
        tweet_model.cuda()
    else:
        tweet_model.cpu()
    # end if

    # Load tweet model vocabulary
    tweet_voc = torch.load(open(os.path.join(tweet_model_dir, "voc_" + lang + ".p"), 'rb'))

    return image_model, tweet_model, tweet_voc
# end load_models

################
# Results
################


# Save results
def save_result(output, author_id, lang, gender_txt, gender_img, gender_both):
    """
    Save results
    :param output:
    :param author_id:
    :param gender_txt:
    :param gender_img:
    :param gender_both:
    :return:
    """
    # File output
    file_output = os.path.join(output, author_id + ".xml")

    # Log
    print(u"Writing result for {} to {}".format(author_id, file_output))

    # Open
    f = codecs.open(os.path.join(output, author_id + ".xml"), 'w', encoding='utf-8')

    # Write
    f.write(u"<author id=\"{}\" lang=\"{}\" gender_txt=\"{}\" gender_img=\"{}\" gender_comb=\"{}\"/>".
            format(
                    author_id,
                    lang,
                    settings.idx_to_class[int(gender_txt[0])],
                    settings.idx_to_class[int(gender_img[0])],
                    settings.idx_to_class[int(gender_both[0])]
            )
    )

    # Close
    f.close()
# end save_result
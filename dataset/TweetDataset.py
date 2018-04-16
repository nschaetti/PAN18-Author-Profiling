#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : dataset/TweetDataset.py
# Description : The PAN17/18 tweet data set for gender profiling.
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
from torch.utils.data.dataset import Dataset
import urllib
import os
import zipfile
from lxml import etree
import codecs
import torch


# The PAN17/18 tweet data set
class TweetDataset(Dataset):
    """
    Tweet dataset
    """

    # Constructor
    def __init__(self, root='./data', download=True, lang='en', text_transform=None, year=2018, train=True, val=0.1):
        """
        Constructor
        :param root: Dataset root directory
        :param download: Do you want to download the dataset?
        :param lang: Which lang to load?
        :param text_transform: Text transformation (from TorchLanguage)
        """
        # Properties
        self.root = os.path.join(root, str(year))
        self.lang = lang
        self.text_transform = text_transform
        self.downloaded = False
        self.classes = {'female': 0, 'male': 1}
        self.year = year
        self.train = train
        self.val = val
        self.max = 0
        self.input_dim = 0
        self.transformed_dim = 0

        # List of author's IDs
        self.idxs = list()

        # Create directory
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Scan root
        for file_name in os.listdir(self.root):
            if u".xml" in file_name:
                self.downloaded = True
                break
            # end if
        # end for

        # Download the data set
        if download and not self.downloaded:
            self._download()
        # end if

        # Load labels
        self.labels = self._load_labels()

        # Load idxs
        self._load()
    # end __init__

    ##############################################
    # OVERRIDE
    ##############################################

    # Length
    def __len__(self):
        """
        Length
        :return: The data set length
        """
        # Total len
        total_length = len(self.idxs)

        # Validation len
        validation_length = int(total_length * self.val)

        # Train length
        train_length = total_length - validation_length

        if self.train:
            return train_length
        else:
            return validation_length
        # end if
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item: The item's index to return
        :return: The corresponding item
        """
        # Total len
        total_length = len(self.idxs)

        # Validation len
        validation_length = int(total_length * self.val)

        # Train length
        train_length = total_length - validation_length

        # Current set
        if self.train:
            current_set = self.idxs[:train_length]
        else:
            current_set = self.idxs[train_length:]
        # end if

        # Current IDXS
        current_idxs = current_set[item]

        # Path to file
        path_to_file = os.path.join(self.root, current_idxs + ".xml")

        # Load  XML
        tree = etree.parse(path_to_file)

        # Texts and images
        tweets = torch.Tensor()

        # Get each documents
        start = True
        for document in tree.xpath("/author/documents/document"):
            # Transformed
            transformed = self.text_transform(unicode(document.text))

            # Add one empty dim
            transformed = transformed.unsqueeze(0)

            # Add
            if start:
                tweets = transformed
                start = False
            else:
                tweets = torch.cat((tweets, transformed), dim=0)
            # end if
        # end for

        # Check number of tweets
        if tweets.size(0) != 100:
            tensor_type = tweets.__class__
            for i in range(100-tweets.size(0)):
                tweets = torch.cat((tweets, tensor_type(1, transformed.size(1)).fill_(0)))
            # end for
        # end if

        # Labels
        labels = torch.LongTensor(tweets.size(0)).fill_(self.labels[current_idxs])

        return tweets, labels
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Create the root directory
    def _create_root(self):
        """
        Create the root directory
        :return:
        """
        # Root
        os.mkdir(self.root)

        # Create year dir
        os.mkdir(os.path.join(self.root, str(self.year)))
    # end _create_root

    # Download the dataset
    def _download(self):
        """
        Download the dataset
        :return:
        """
        # Filename
        if self.year == 2017:
            zip_filename = "pan17-author-profiling.zip"
        elif self.year == 2018:
            zip_filename = "pan18-author-profiling.zip"
        # end if

        # Path to zip file
        path_to_zip = os.path.join(self.root, zip_filename)

        # Download
        if not os.path.exists(path_to_zip):
            print(u"Downloading {}".format("http://www.nilsschaetti.com/datasets/" + zip_filename))
            urllib.urlretrieve("http://www.nilsschaetti.com/datasets/" + zip_filename, path_to_zip)
        # end if

        # Unzip
        print(u"Unziping {}".format(path_to_zip))
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    # Load labels
    def _load_labels(self):
        """
        Load labels
        :return:
        """
        # Read file
        label_file = codecs.open(os.path.join(self.root, self.lang + ".txt"), encoding='utf-8').read()

        # IDX to labels
        idx_to_labels = {}

        # For each line
        for line in label_file.split("\n"):
            if len(line) > 0:
                # ID and label
                if self.year == 2018:
                    idx, label = line.split(":::")
                elif self.year == 2017:
                    idx, label, _ = line.split(":::")
                # end if

                # Save
                idx_to_labels[idx] = self.classes[label]
            # end if
        # end for

        return idx_to_labels
    # end _load_labels

    # Load dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # For each file
        for file_name in os.listdir(self.root):
            if u".xml" in file_name:
                # Path to the file
                path_to_file = os.path.join(self.root, file_name)

                # Load  XML
                tree = etree.parse(path_to_file)

                # Author
                author = tree.xpath("/author")[0]

                # IDXS
                idxs = file_name[:-4]

                # Check lang
                if author.get("lang") == self.lang:
                    self.idxs.append(idxs)
                # end if
            # end if
        # end for
    # end _load

# end TweetDataset

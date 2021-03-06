# -*- coding: utf-8 -*-
#

# Imports
from torch.utils.data.dataset import Dataset
import urllib
import os
import zipfile
import codecs
from lxml import etree
import json
import codecs
from PIL import Image
import torch


# Author profiling image dataset
class ImageDataset(Dataset):
    """
    Author profiling image dataset
    """

    # Constructor
    def __init__(self, root='./data', download=True, image_transform=None, image_size=600, train=True, val=0.1):
        """
        Constructor
        :param root:
        :param download:
        :param lang:
        :param image_transform:
        """
        # Properties
        self.root = os.path.join(root, "2018")
        self.image_transform = image_transform
        self.downloaded = False
        self.classes = {'female': 0, 'male': 1}
        self.image_size = image_size
        self.train = train
        self.val = val

        # Image list
        self.images = list()

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
        :return:
        """
        # Total len
        total_length = len(self.images)

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
        :param item:
        :return:
        """
        # Total len
        total_length = len(self.images)

        # Validation len
        validation_length = int(total_length * self.val)

        # Train length
        train_length = total_length - validation_length

        # Current set
        if self.train:
            current_set = self.images[:train_length]
        else:
            current_set = self.images[train_length:]
        # end if

        # Current image path
        current_image_path = os.path.join(self.root, current_set[item])

        # Corresponding ID
        idx = current_set[item][:current_set[item].find('.')]

        # PIL image
        try:
            im = Image.open(current_image_path)
        except IOError:
            im = Image.new('RGB', (10, 10))
        # end try

        # Transformed
        transformed_image = self.image_transform(im)

        # Remove alpha
        if transformed_image.size(0) == 4:
            transformed_image = transformed_image[0:3]
        elif transformed_image.size(0) == 1:
            transformed_image = torch.cat((transformed_image, transformed_image, transformed_image), dim=0)
        # end if

        return transformed_image, self.labels[idx]
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
        os.mkdir(self.root)
    # end _create_root

    # Download the dataset
    def _download(self):
        """
        Download the dataset
        :return:
        """
        # Path to zip file
        path_to_zip = os.path.join(self.root, "pan18-author-profiling.zip")

        # Download
        if not os.path.exists(path_to_zip):
            print(u"Downloading {}".format("http://www.nilsschaetti.com/datasets/pan18-author-profiling.zip"))
            urllib.urlretrieve("http://www.nilsschaetti.com/datasets/pan18-author-profiling.zip", path_to_zip)
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
        # IDX to labels
        idx_to_labels = {}

        # For each lang
        for lang in ['en', 'ar', 'es']:
            # Read file
            label_file = codecs.open(os.path.join(self.root, lang + ".txt")).read()

            # For each line
            for line in label_file.split("\n"):
                if len(line) > 0:
                    # ID and label
                    idx, label = line.split(":::")

                    # Save
                    idx_to_labels[idx] = self.classes[label]
                # end if
            # end for
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
            if u".jpeg" in file_name or u".png" in file_name or u".jpg" in file_name or u".gif" in file_name:
                # Add
                self.images.append(file_name)
            elif u".xml" in file_name:
                # IDXS
                idxs = file_name[:-4]

                # Check lang
                self.idxs.append(idxs)
            # end if
        # end for
    # end _load

# end AuthorProfilingDataset

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : dataset/AuthorProfilingDataset.py
# Description : The PAN18 author profiling dataset.
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
import os
from lxml import etree
import codecs
from PIL import Image, ImageFile
import torch


# Avoid issue with truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# TIRA Author profiling data set
class TIRAAuthorProfilingDataset(Dataset):
    """
    TIRA Author profiling dataset
    """

    # Constructor
    def __init__(self, root, lang, text_transform=None, image_transform=None, filter_robot=False):
        """
        Constructor
        :param root: Data root directory
        :param lang: Which lang to use
        :param text_transform: Text transformation (from TorchLanguage)
        :param image_transform: Image transformation (from TorchVision)
        """
        # Properties
        self.root = os.path.join(root, lang)
        self.lang = lang
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.classes = {'female': 0, 'male': 1}
        self.last_idxs = list()
        self.filter_robot = filter_robot

        # Load labels
        self.labels, self.idxs = self._load_labels()

        # Load idxs
        # self._load()
    # end __init__

    ##############################################
    # OVERRIDE
    ##############################################

    # Length
    def __len__(self):
        """
        Length
        :return: Data set length
        """
        # Total len
        return len(self.idxs)
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item: Which item to return
        :return: The item
        """
        # Current IDXs
        current_idxs = self.idxs[item]
        self.last_idxs.append(current_idxs)

        # Directories
        photo_directory = os.path.join(self.root, "photo")
        text_directory = os.path.join(self.root, "text")

        # Path to file
        path_to_file = os.path.join(text_directory, current_idxs + ".xml")

        # Load  XML
        tree = etree.parse(path_to_file)

        # Texts and images
        tweets = torch.Tensor()
        images = torch.Tensor()

        # Get each documents
        start = True
        for document in tree.xpath("/author/documents/document"):
            if not self.filter_robot or u"// automatically checked by" not in document.text:
                # Transformed
                transformed = self.text_transform(document.text)

                # Add one empty dim
                transformed = transformed.unsqueeze(0)

                # Add
                if start:
                    tweets = transformed
                    start = False
                else:
                    tweets = torch.cat((tweets, transformed), dim=0)
                # end if
            # end if
        # end for

        # Get each images
        start = True
        for i in range(10):
            # Image path
            image_path_jpeg = os.path.join(photo_directory, current_idxs, current_idxs + "." + str(i) + ".jpeg")
            image_path_png = os.path.join(photo_directory, current_idxs, current_idxs + "." + str(i) + ".png")

            # Check existence
            if os.path.exists(image_path_jpeg):
                image_path = image_path_jpeg
            else:
                image_path = image_path_png
            # end if

            # PIL image
            try:
                im = Image.open(image_path)
            except IOError:
                print(u"IOError while loading {}".format(image_path))
                im = Image.new('RGB', (10, 10))
            # end try

            # Transformed
            transformed_image = self.image_transform(im)

            # Add empty dim
            transformed_image = transformed_image.unsqueeze(0)

            # Remove alpha
            if transformed_image.size(1) == 4:
                transformed_image = transformed_image[:, 0:3]
            elif transformed_image.size(1) == 1:
                transformed_image = torch.cat((transformed_image, transformed_image, transformed_image), dim=1)
            # end if

            # Add image
            if start:
                images = transformed_image
                start = False
            else:
                images = torch.cat((images, transformed_image), dim=0)
            # end if
        # end for

        return tweets, images, self.labels[current_idxs]
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Create the root directory
    def _create_root(self):
        """
        Create the root directory.
        """
        os.mkdir(self.root)
    # end _create_root

    # Load labels
    def _load_labels(self):
        """
        Load labels
        :return: Dictionary from id to labels
        """
        # Read file
        label_file = codecs.open(os.path.join(self.root, "truth.txt")).read()

        # IDX to labels
        idx_to_labels = {}
        idxs = list()

        # For each line
        for line in label_file.split("\n"):
            if len(line) > 0:
                # ID and label
                idx, label = line.split(":::")

                # Save label and idx
                idx_to_labels[idx] = self.classes[label]
                idxs.append(idx)
            # end if
        # end for

        return idx_to_labels, idxs
    # end _load_labels

# end TIRAAuthorProfilingDataset

# -*- coding: utf-8 -*-
#

# Imports
from torch.utils.data.dataset import Dataset
import urllib
import os
import zipfile
from lxml import etree
import json
import codecs


# Author profiling data set
class AuthorProfilingDataset(Dataset):
    """
    Author profiling dataset
    """

    # Constructor
    def __init__(self, root='./data', download=True, lang='en', text_transform=None, image_transform=None):
        """
        Constructor
        :param root:
        :param download:
        :param lang:
        :param text_transform:
        :param image_transform:
        """
        # Properties
        self.root = root
        self.lang = lang
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.downloaded = False

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
        return len(self.idxs)
    # end __len__

    # Get item
    def __getitem__(selfs, item):
        """
        Get item
        :param item:
        :return:
        """
        pass
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
            urllib.urlretrieve("http://www.nilsschaetti.com/datasets/pan18-author-profiling.zip", path_to_zip)
        # end if

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

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
                print(tree.xpath("/author"))
                print(tree.xpath("/author").get("lang"))
                print(tree.xpath("/author/documents"))
                exit()
                # Author
                print(tree.author.lang)
            # end if
        # end for
    # end _load

# end AuthorProfilingDataset

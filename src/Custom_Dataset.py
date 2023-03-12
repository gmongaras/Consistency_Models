from torch.utils.data import Dataset
import torchvision
import torch
from helpers import reduce_image
import pickle
import os
import numpy as np
import math
from copy import deepcopy




class CustomDataset(Dataset):
    """Generative Dataset."""

    def __init__(self, data_type, shuffle=True, loadMem=False, train=True):
        """
        Args:
            data_type (str): Type of data to load in (MNIST, ...)
            shuffle (boolean): True to shuffle the data upon entering. False otherwise
            loadMem (boolean): True to load in all data to memory, False to keep it on disk
            train (boolean): True to load in train data, False to load in test data
        """

        # Save the data information
        self.loadMem = loadMem

        # Used to transform the data to a transor
        self.transform = torchvision.transforms.Compose(
            [
                # Transform to a tensor
                torchvision.transforms.ToTensor(),
            ]
        )

        # Load the dataset
        if data_type == "MNIST":
            self.dataset = torchvision.datasets.MNIST("./data/", train=train, transform=self.transform, download=True)
        else:
            raise NotImplementedError
        
        self.num_data = len(self.dataset)

        
        # Create a list of indices which can be used to
        # essentially shuffle the data
        self.data_idxs = np.arange(0, self.num_data)
        if shuffle:
            np.random.shuffle(self.data_idxs)


        
    def __len__(self):
        return self.num_data

    # idx - Index of the data to get (can be a single index of multiple indices)
    def __getitem__(self, idx):

        # Convert the given index to the shuffled index
        data_idx = self.data_idxs[idx]

        # Get the data from the dataset
        X = self.dataset.data[data_idx].clone()
        cls = self.dataset.targets[data_idx].clone()
        if len(X.shape) == 2 and type(idx) == int:
            X = X.unsqueeze(0)
        elif len(X.shape) == 3 and type(idx) != int:
            X = X.unsqueeze(1)

        # Transform the image between -1 and 1
        X = reduce_image(X)

        return X, cls
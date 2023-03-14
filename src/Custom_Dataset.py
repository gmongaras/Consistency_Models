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
        elif data_type == "CIFAR10":
            self.dataset = torchvision.datasets.CIFAR10("./data/", train=train, transform=self.transform, download=True)
        elif data_type == "CIFAR100":
            self.dataset = torchvision.datasets.CIFAR100("./data/", train=train, transform=self.transform, download=True)
        else:
            raise NotImplementedError
        
        # Load the classes as np arrays
        self.dataset.targets = np.array(self.dataset.targets)
        
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
        try:
            X = self.dataset.data[data_idx].clone()
        except AttributeError:
            X = torch.tensor(self.dataset.data[data_idx])
        try:
            cls = self.dataset.targets[data_idx].clone()
        except AttributeError:
            cls = torch.tensor(self.dataset.targets[data_idx])
        if len(X.shape) == 2 and type(idx) == int:
            X = X.unsqueeze(0)
        elif len(X.shape) == 3 and type(idx) != int:
            X = X.unsqueeze(1)

        # Shape correction
        if X.shape[0] > 3 and len(X.shape) == 3:
            X = X.permute(2, 0, 1)
        elif X.shape[1] > 3 and len(X.shape) == 4:
            X = X.permute(0, 3, 1, 2)

        # Transform the image between -1 and 1
        X = reduce_image(X)

        return X, cls
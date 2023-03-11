from torch.utils.data import Dataset
import torchvision
import torch
from helpers import reduce_image
import pickle
import os
import numpy as np
import math




class CustomDataset(Dataset):
    """Generative Dataset."""

    def __init__(self, data_type, shuffle=True, loadMem=False):
        """
        Args:
            data_type (str): Type of data to load in (MNIST, ...)
            shuffle (boolean): True to shuffle the data upon entering. False otherwise
            loadMem (boolean): True to load in all data to memory, False to keep it on disk
        """

        # Save the data information
        self.loadMem = loadMem

        # Used to transform the data to [-1, 1]
        self.transform = torchvision.transforms.Compose(
            [
                # Transform to a tensor
                torchvision.transforms.ToTensor(),
                # Transform between -1 and 1
                reduce_image
            ]
        )

        # Load the dataset
        if data_type == "MNIST":
            self.dataset = torchvision.datasets.MNIST("./", train=True, transform=self.transform, download=True)
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

    def __getitem__(self, idx):

        # Convert the given index to the shuffled index
        data_idx = self.data_idxs[idx]

        # Get the data from the dataset and return it
        return self.dataset[data_idx]
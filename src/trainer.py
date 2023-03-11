import torch
from torch import nn
from torch import optim
import torchvision
from helpers import reduce_image








class trainer():
    def __init__(self, model):
        # Save the model
        self.model = model

        # Optimizer for the model
        self.optim = optim.AdamW(self.model.parameters())


        # Used to transform the data to [-1, 1]
        transform = torchvision.transforms.Compose(
            [
                # Transform to a tensor
                torchvision.transforms.ToTensor(),
                # Transform between -1 and 1
                reduce_image
            ]
        )

        # Create the dataset object
        dataset = torchvision.datasets.MNIST("./", train=True, transform=transform, download=True)

    def train(self):
        pass

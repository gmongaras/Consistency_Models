import torch
from torch import nn
from torch import optim
import torchvision
from helpers import reduce_image
from copy import deepcopy
from Custom_Dataset import CustomDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader








class Trainer():
    def __init__(self, config, model):
        self.device = config.device
        self.mu = torch.tensor(config.training.mu, device=self.device)
        self.mu_0 = torch.tensor(config.training.mu_0, device=self.device)
        self.s_0 = torch.tensor(config.training.s_0, device=self.device)
        self.s_1 = torch.tensor(config.training.s_1, device=self.device)
        self.epsilon = torch.tensor(config.model.epsilon, device=self.device)
        self.t_rho = torch.tensor(config.model.t_rho, device=self.device)
        self.n_iters = config.training.n_iters
        self.K = torch.tensor(self.n_iters, device=self.device)
        self.batch_size = config.training.batch_size
        loss_funct = config.training.loss_funct

        # Save the model
        self.model_norm = model

        # Copy the model to a second version of itself
        self.model_minus = deepcopy(model)

        # Optimizer for the model
        self.optim = optim.AdamW(self.model_norm.parameters())

        # Create the dataset object
        self.dataset = CustomDataset(config.dataset.data_type, config.dataset.shuffle, config.dataset.loadMem)

        # Unit normal distribution
        self.unit_normal = torch.distributions.normal.Normal(0, 1)

        # Loss function
        if loss_funct == "l1":
            self.loss_funct = torch.nn.L1Loss(reduction="mean")
        elif loss_funct == "l2":
            self.loss_funct == torch.nn.MSELoss(reduction="mean")
        else:
            raise NotImplementedError


    # Used to calculate the N(k) scheduling funciton
    def N_sched(self, k):
        return torch.ceil(
            torch.sqrt(
                (k/self.K)*((self.s_1+1)**2 - self.s_0**2) + self.s_0**2
            ) - 1
        ) + 1
    
    # Used to calcualte the mu(k) scheduling function
    def mu_sched(self, k):
        return torch.exp(
            (self.s_0*torch.log(self.mu_0))/\
                self.N_sched(k)
        )


    def train(self):
        # Initialize k (for the scheduler) to 0. This is the number of update steps
        numSteps = 0

        # Create a sampler and loader over the dataset
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                pin_memory=False, num_workers=2, 
                drop_last=False, shuffle=True
            )

        # Iterate until we've reached the number of desired steps
        while numSteps < self.n_iters:
            # Iterate over the dataset
            for step, data in enumerate(data_loader):
                k = numSteps

                # Get the data and label (x~D)
                X, label = data

                X = X.to(self.device)

                # Sample values of n uniformly (n~U)
                if self.N_sched(k)-1 == 1:
                    n_samp = torch.ones(X.shape[0], device=self.device)
                else:
                    n_samp = torch.distributions.uniform.Uniform(1, self.N_sched(k)-1).sample((X.shape[0],)).to(self.device)

                # Convert the n samples to timesteps t_n and t_n+1
                timesteps = self.model_norm.n_to_t(n_samp, self.N_sched(k))
                timesteps_1 = self.model_norm.n_to_t(n_samp+1, self.N_sched(k))

                # Sample noise from a unit normal distribtuion
                noise = self.unit_normal.sample(X.shape).to(self.device)

                # Forward through the model normal model
                model_norm_out = self.model_norm(X+timesteps_1.reshape(-1, 1, 1, 1)*noise, timesteps_1)

                # Forward through the second model. Note that
                # we don't need any gradients for this model
                with torch.no_grad():
                    model_minus_out = self.model_minus(X+timesteps.reshape(-1, 1, 1, 1)*noise, timesteps)

                # Get the loss for the first model
                loss_model_norm = self.loss_funct(model_norm_out, model_minus_out.detach())

                # Update the first model using SGD
                loss_model_norm.backward()
                self.optim.step()
                self.optim.zero_grad()

                # Update the second model by 
                # mu(k)*model_minus_params + (1-mu(k))*model_norm_params
                mu_k = self.mu_sched(k)
                model_norm_params = self.model_norm.state_dict()
                model_minus_params = self.model_minus.state_dict()
                for param in model_minus_params.keys():
                    model_norm_param = model_norm_params[param]
                    model_minus_param = model_minus_params[param]
                    model_minus_params[param] = mu_k*model_minus_param + (1-mu_k)*model_norm_param
                self.model_minus.load_state_dict(model_minus_params)

                # Increase the number of steps
                numSteps += 1

            print(loss_model_norm)

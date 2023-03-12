import torch
from torch import nn
from score_sde.models.ncsnpp import NCSNpp
import numpy as np






class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # Save parts of the given config file
        self.device = config.device
        self.sigma_data = torch.tensor(config.model.sigma_data, device=self.device)
        self.t_rho = torch.tensor(config.model.t_rho, device=self.device)
        self.max_T = torch.tensor(config.model.max_T, device=self.device)
        self.epsilon = torch.tensor(config.model.epsilon, device=self.device)

        # Create the nscp++ model
        self.NCSNpp = NCSNpp(config).to(self.device)




    # Used to calculate the c_skip function
    def c_skip(self, t):
        return ((self.sigma_data**2)/\
                ((t-self.epsilon)**2 + self.sigma_data**2)).reshape(-1, 1, 1, 1)

    # Used to calculate the c_out function
    def c_out(self, t):
        return ((self.sigma_data*(t-self.epsilon))/\
                torch.sqrt(self.sigma_data**2 + t**2)).reshape(-1, 1, 1, 1)
    
    
   # Convert a value of n to a value of t (EQ 5.5?? right above EQ 6)
    # n - Value to convert to t
    # N - Upper bound on the value of n
    def n_to_t(self, n, N):
        return \
            (self.epsilon**(1/self.t_rho)\
                + ((n-1)/(N-1))\
                * (self.max_T**(1/self.t_rho) - self.epsilon**(1/self.t_rho))
            )**self.t_rho


    # Forward function
    # Input:
    #   X - Image batch of shape (N, C, L, W)
    #   t - Time conditional of shape (N)
    #   cls - Class labels of shape (N)
    # Output:
    #   Image batch of shape (N, C, L, W)
    def forward(self, X, t, cls=None):
        # Forward pass throug the NCSN++ model (F(x, t))
        F_xt = self.NCSNpp(X, t)

        # Construct the output (EQ 5): 
        # f(x, t) = c_skip(t)x + c_out(t)F(x, t)
        f_xt = self.c_skip(t)*X + self.c_out(t)*F_xt

        return f_xt.to(torch.float32)
    


    # Used to sample an image from the current model
    # shape - Shape of the image. Ex: (3, 64, 64)
    # N - Number of denoising steps in the sampling trajectory
    # batch_size - Number of images to generate in parallel
    def sample(self, shape, N, batch_size):
        assert N >= 1 and N <= self.max_T, \
            "The number of sampling steps (sample_N) must be within [1, max_T]"

        # Sample initial image: X_hat_T ~ N(0, T**2)
        x = torch.distributions.normal.Normal(0, self.max_T.cpu().item()**2).sample([batch_size] + shape).to(self.device)

        # Put the initial image through the model
        x = self.forward(x, torch.repeat_interleave(self.max_T.unsqueeze(0), batch_size, dim=0))

        # Clamp x between -1 and 1
        x = x.clamp(-1, 1)

        # Unit normal distribution
        unit_normal = torch.distributions.normal.Normal(0, 1)

        #### NOTE: The model work based on what step it's on, not
        #### what step it's going to. So if it needs to go from
        #### step T to step 1, you give it T. IT is trained to
        #### go from any step back to step 1 so giving it noise
        #### and going back to step 1 may help it a little, but
        #### isn't necessary!

        # Iterate from n=N-1 to n=2 which representnts decreasing
        # values of t from max_T to 1. When N = max_T, this
        # sequence decreases by 1, else this is a subset. 
        # Note: We don't have to run n=1 since that doesn't change the output
        #       due to the properties of c_skip and c_out
        for n in range(int(N.cpu().item())-1, 1, -1):
            # Sample from a unit normal distribution
            z = unit_normal.sample([batch_size] + shape).to(x.device)

            # Map the n index to the value of t
            t = self.n_to_t(torch.tensor([n], device=x.device), N)
            t = torch.repeat_interleave(t, batch_size, dim=0)
            t_exten = t.reshape(-1, 1, 1, 1)
            x = x + torch.sqrt(torch.max(torch.tensor(0), t_exten**2 - self.epsilon**2))*z

            # Update x using the model
            x = self.forward(x.to(torch.float32), t)

            # Clamp x between -1 and 1
            x = x.clamp(-1, 1)
        
        # Return the sample
        return x.to(torch.float32)
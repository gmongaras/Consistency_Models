import torch
from torch import nn
from score_sde.models.ncsnpp import NCSNpp






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

        return f_xt
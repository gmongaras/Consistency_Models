from Model import Model
from Trainer import Trainer
from ncspp_configs import get_MNIST_configs
import torch
import ml_collections






def main():
    # Extra parameters
    lr = 4e-4
    batch_size = 256
    eval_batch_size = 64
    sample_size = 10
    sample_steps = 100
    sample_N = 5 # In the sample trajectory, how many steps to take to generate the image (between 1 and max_T)
    EMA_decay_rate = 0.9999
    n_iters = 100000
    loss_funct = "l2" # L1, L2, or LIPS for loss

    # Consistency model parameters
    sigma_data = 0.5
    mu = 0
    mu_0 = 0.9
    s_0 = 2
    s_1 = 150
    epsilon = 0.002
    t_rho = 7
    max_T = 80

    # Dataset options
    data_type = "MNIST"
    shuffle = True
    loadMem = True




    # Get the configuration file for the ncsp++ model
    config = get_MNIST_configs()

    # Change some configs for the new parameters
    config.optim.lr = lr
    config.training.batch_size = batch_size
    config.eval.batch_size = eval_batch_size
    config.sampling.sample_size = sample_size
    config.sampling.sample_steps = sample_steps
    config.training.n_iters = n_iters
    config.training.loss_funct = loss_funct
    config.training.sample_N = sample_N

    config.model.sigma_data = sigma_data
    config.model.t_rho = t_rho
    config.model.max_T = max_T
    config.training.mu = mu
    config.training.mu_0 = mu_0
    config.training.s_0 = s_0
    config.training.s_1 = s_1
    config.model.epsilon = epsilon

    config.dataset = ml_collections.ConfigDict({})
    config.dataset.data_type = data_type
    config.dataset.shuffle = shuffle
    config.dataset.loadMem = loadMem

    # Create the model
    model = Model(config)

    # Create the model trainer
    trainer = Trainer(config, model)

    # Train the model
    trainer.train2(config, "./outputs")







if __name__ == "__main__":
    main()
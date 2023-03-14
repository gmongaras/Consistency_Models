from Model import Model
from Trainer import Trainer
from ncspp_configs import get_MNIST_configs, get_CIFAR10_configs
import torch
import ml_collections
import json
import os






def main():
    # Extra parameters
    lr = 4e-4
    batch_size = 128
    eval_batch_size = 64
    sample_size = 10
    sample_steps = 100
    sample_N = 1 # In the sample trajectory, how many steps to take to generate the image (between 1 and max_T)
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
    data_type = "CIFAR10"
    shuffle = True
    loadMem = True

    # Saving
    baseDir = "outputs/CIFAR10"
    save_dir = "outputs/CIFAR10/saved_models"

    # Loading
    loadModel = True
    load_dir = "outputs/CIFAR10/saved_models/iter_89000"
    load_file_norm = "model_norm_89000s.pt"
    load_file_minus = "model_minus_89000s.pt"
    load_file_optim = "optim_89000s.pt"
    load_file_config = "config_89000s.json"




    # Get the configuration file for the ncsp++ model if
    # a model isn't going to be loaded. Otherwise, load the
    # configuration file
    if not loadModel:
        config = get_CIFAR10_configs()

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

        config.saving = ml_collections.ConfigDict({})
        config.saving.save_dir = save_dir

        config.loading = ml_collections.ConfigDict({})
        config.loading.loadModel = loadModel
    
    else:
        # Open and load in the config file
        with open(load_dir + os.sep + load_file_config, "r") as f:
            config = json.load(f)
        config = ml_collections.ConfigDict(config)

        # Add important information to the config file
        config.device = config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        config.loading = ml_collections.ConfigDict({})
        config.loading.loadModel = loadModel
        config.loading.load_dir = load_dir
        config.loading.load_file_norm = load_file_norm
        config.loading.load_file_minus = load_file_minus
        config.loading.load_file_optim = load_file_optim
        config.loading.load_file_config = load_file_config

        # Overwriting
        config.training.sample_N = sample_N

    # Create the model
    model = Model(config)

    # Create the model trainer
    trainer = Trainer(config, model)

    # Train the model
    trainer.train(config, baseDir)







if __name__ == "__main__":
    main()
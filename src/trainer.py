import torch
from torch import nn
from torch import optim
import torchvision
from helpers import reduce_image
from copy import deepcopy
from Custom_Dataset import CustomDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LIPS



import os
import numpy as np
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from score_sde.models import ddpm, ncsnv2, ncsnpp
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import json








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
        self.loss_funct = config.training.loss_funct
        self.num_ch = config.data.num_channels
        self.sample_N = torch.tensor(config.training.sample_N, device=self.device)
        self.save_dir = config.saving.save_dir

        # Remove the device from the config file to
        # help with saving the file
        if "device" in config:
            del config.device

        # Save the config file
        self.config = config

        # Save the model
        self.model_norm = model

        # Copy the model to a second version of itself
        self.model_minus = deepcopy(model)

        # Optimizer for the model
        self.optim = optim.AdamW(self.model_norm.parameters())


        # If models should be loaded, load in the models
        if self.config.loading.loadModel == True:
            self.load_models()


        # Create the dataset objects for train and eval
        self.dataset_train = CustomDataset(config.dataset.data_type, config.dataset.shuffle, config.dataset.loadMem, train=True)
        self.dataset_eval = CustomDataset(config.dataset.data_type, config.dataset.shuffle, config.dataset.loadMem, train=False)

        # Unit normal distribution
        self.unit_normal = torch.distributions.normal.Normal(0, 1)

        # Loss function
        if self.loss_funct == "l1":
            self.loss_funct = torch.nn.L1Loss(reduction="mean")
        elif self.loss_funct == "l2":
            self.loss_funct = torch.nn.MSELoss(reduction="mean")
        elif self.loss_funct == "LIPS":
            self.loss_funct = LIPS(net_type="alex", reduction="mean", normalize=False)
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

    











    # Used to step the model once
    def train_step_fn(self, data, k):
        # Get the data and label (x~D)
        X, label = data

        X = X.to(self.device)

        # Sample values of n uniformly (n~U)
        if self.N_sched(k)-1 == 1:
            n_samp = torch.ones(X.shape[0], device=self.device)
        else:
            n_samp = torch.distributions.uniform.Uniform(1, self.N_sched(k)-1).sample((X.shape[0],)).to(self.device)

        adds = []
        for i in range(0, X.shape[0]):
            if self.N_sched(k)-1 == 1:
                adds.append(torch.tensor(1).to(self.device))
            else:
                adds.append(
                    torch.distributions.uniform.Uniform(1, self.N_sched(k)-n_samp[i]).sample((1,)).to(self.device)
                )
        adds = torch.stack(adds).squeeze()

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

        # If the number of channels is 1, repeat it to 3.
        # if LIPS loss is being used
        if self.num_ch == 1 and self.loss_funct == "LIPS":
            model_norm_out = torch.repeat_interleave(model_norm_out, 3, dim=1)
            model_minus_out = torch.repeat_interleave(model_minus_out, 3, dim=1)

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

        return loss_model_norm
    

    @torch.no_grad()
    def eval_fn(self, data, k):
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

        # Forward through the normal model
        model_norm_out = self.model_norm(X+timesteps_1.reshape(-1, 1, 1, 1)*noise, timesteps_1)

        # Forward through the second model. Note that
        # we don't need any gradients for this model
        model_minus_out = self.model_minus(X+timesteps.reshape(-1, 1, 1, 1)*noise, timesteps)

        # If the number of channels is 1, repeat it to 3.
        # if LIPS loss is being used
        if self.num_ch == 1 and self.loss_funct == "LIPS":
            model_norm_out = torch.repeat_interleave(model_norm_out, 3, dim=1)
            model_minus_out = torch.repeat_interleave(model_minus_out, 3, dim=1)

        # Get the loss for the first model
        loss_model_norm = self.loss_funct(model_norm_out, model_minus_out.detach())

        return loss_model_norm









    def train(self, config, workdir):
        """Runs the training pipeline.

        Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
            contains checkpoint training will be resumed from the latest checkpoint.
        """

        # Create directories for experimental logs
        sample_dir = os.path.join(workdir, "samples")
        tf.io.gfile.makedirs(sample_dir)

        tb_dir = os.path.join(workdir, "tensorboard")
        tf.io.gfile.makedirs(tb_dir)
        writer = tensorboard.SummaryWriter(tb_dir)

        # Create a train sampler and loader over the dataset
        data_loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size,
                pin_memory=False, num_workers=2, 
                drop_last=False, shuffle=True
            )

        # Variable initializations
        num_train_steps = config.training.n_iters
        try:
            totalSteps = self.step_start
        except AttributeError:
            totalSteps = 0

        # In case there are multiple hosts (e.g., TPU pods), only log to host 0
        logging.info("Starting training loop at step %d." % (totalSteps,))
        print("Starting training loop at step %d." % (totalSteps,))

        for step in range(totalSteps, num_train_steps + 1):
            # Iterate over the dataset
            for step, data in enumerate(data_loader_train):
                k = totalSteps

                loss_model_norm = self.train_step_fn(data, k)

                # Increase the number of steps
                totalSteps += 1




                




                
                loss = loss_model_norm

                # Log the current raining loss every so often
                if totalSteps % config.training.log_freq == 0:
                    logging.info("step: %d, training_loss: %.5e" % (totalSteps, loss.item()))
                    print("step: %d, training_loss: %.5e" % (totalSteps, loss.item()))
                    writer.add_scalar("training_loss", loss, totalSteps)

                # Report the loss on an evaluation dataset periodically
                if totalSteps % config.training.eval_freq == 0:
                    with torch.no_grad():
                        # Sample a random batch of eval data
                        random_idx = torch.tensor(np.random.choice(len(self.dataset_eval), config.eval.batch_size, replace=False), dtype=torch.long)
                        eval_batch = self.dataset_eval[random_idx]
                        
                        # Evaluate the model
                        eval_loss = self.eval_fn(eval_batch, k)

                        # Log the eval loss
                        logging.info("step: %d, eval_loss: %.5e" % (totalSteps, eval_loss.item()))
                        print("step: %d, eval_loss: %.5e" % (totalSteps, eval_loss.item()))
                        writer.add_scalar("eval_loss", eval_loss.item(), totalSteps)

                        del eval_batch
                        del eval_loss

                # Save a checkpoint periodically and generate samples if needed
                if totalSteps != 0 and totalSteps % config.training.snapshot_freq == 0 or totalSteps == num_train_steps:
                    # Save the checkpoint.
                    self.save_models(os.path.join(self.save_dir, "iter_{}".format(totalSteps)), step=totalSteps)

                # Generate and save samples
                if totalSteps % config.sampling.sample_steps == 0:
                    with torch.no_grad():
                        # Sample from the model
                        shape = list(data[0].shape)[1:]
                        sample = self.model_norm.sample(shape, self.sample_N, config.sampling.sample_size)

                        # Setup the directories
                        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(totalSteps))
                        tf.io.gfile.makedirs(this_sample_dir)

                        # Create the image grid
                        nrow = int(np.sqrt(sample.shape[0]))
                        image_grid = make_grid(sample, nrow, padding=2)

                         # Save the file
                        with tf.io.gfile.GFile(
                                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                            save_image(image_grid, fout)




    # Save the models
    # saveDir - Directory to save the model state to
    # epoch (optional) - Current epoch of the model (helps when loading state)
    # step (optional) - Current step of the model (helps when loading state)
    def save_models(self, saveDir, epoch=None, step=None):
        # Craft the save string
        saveFile_norm = "model_norm"
        saveFile_minus = "model_minus"
        optimFile = "optim"
        saveDefFile = "config"
        if epoch:
            saveFile_norm += f"_{epoch}e"
            saveFile_minus += f"_{epoch}e"
            optimFile += f"_{epoch}e"
            saveDefFile += f"_{epoch}e"
        if step:
            saveFile_norm += f"_{step}s"
            saveFile_minus += f"_{step}s"
            optimFile += f"_{step}s"
            saveDefFile += f"_{step}s"
        saveFile_norm += ".pt"
        saveFile_minus += ".pt"
        optimFile += ".pt"
        saveDefFile += ".json"

        # Change epoch and step state if given
        if epoch:
            self.config.epoch = epoch
        if step:
            self.config.step = step
        
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the models and the optimizer
        torch.save(self.model_norm.state_dict(), saveDir + os.sep + saveFile_norm)
        torch.save(self.model_minus.state_dict(), saveDir + os.sep + saveFile_minus)
        torch.save(self.optim.state_dict(), saveDir + os.sep + optimFile)

        # Save the config file
        with open(saveDir + os.sep + saveDefFile, "w") as f:
            json.dump(self.config.to_dict(), f)

    

    # Load models and optimizer
    def load_models(self):
        # Load in both the normal and minus models
        self.model_norm.loadModel(self.config.loading.load_dir, self.config.loading.load_file_norm)
        self.model_minus.loadModel(self.config.loading.load_dir, self.config.loading.load_file_minus)

        # Load in the optimizer
        self.optim.load_state_dict(torch.load(self.config.loading.load_dir + os.sep + self.config.loading.load_file_optim, map_location=self.device))

        # Load in the number of steps
        if "step" in self.config:
            self.step_start = self.config.step + 1
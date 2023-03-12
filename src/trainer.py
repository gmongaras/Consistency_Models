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

        # Save the model
        self.model_norm = model

        # Copy the model to a second version of itself
        self.model_minus = deepcopy(model)

        # Optimizer for the model
        self.optim = optim.AdamW(self.model_norm.parameters())

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


    def train(self):
        # Initialize k (for the scheduler) to 0. This is the number of update steps
        numSteps = 0

        # Create a sampler and loader over the dataset
        data_loader = DataLoader(self.dataset_train, batch_size=self.batch_size,
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









    def train2(self, config, workdir):
        """Training and evaluation for score-based generative models. """

        import gc
        import io
        import os
        import time

        import numpy as np
        import tensorflow as tf
        import tensorflow_gan as tfgan
        import logging
        # Keep the import below for registering all model definitions
        from score_sde.models import ddpm, ncsnv2, ncsnpp
        import score_sde.losses as losses
        import score_sde.sampling as sampling
        from score_sde.models import utils as mutils
        from score_sde.models.ema import ExponentialMovingAverage
        import score_sde.datasets as datasets
        import score_sde.evaluation as evaluation
        import score_sde.likelihood as likelihood
        import score_sde.sde_lib as sde_lib
        from absl import flags
        import torch
        from torch.utils import tensorboard
        from torchvision.utils import make_grid, save_image
        from score_sde.utils import save_checkpoint, restore_checkpoint

        FLAGS = flags.FLAGS











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

        # Initialize model.
        score_model = mutils.create_model(config)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Create checkpoints directory
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        # Intermediate checkpoints to resume training after pre-emption in cloud environments
        checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
        tf.io.gfile.makedirs(checkpoint_dir)
        tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
        # Resume training when intermediate checkpoints are detected
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        initial_step = int(state['step'])

        # # Build data iterators
        # train_ds, eval_ds, _ = datasets.get_dataset(config,
        #                                             uniform_dequantization=config.data.uniform_dequantization)
        # train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
        # eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
        # # Create data normalizer and its inverse
        # scaler = datasets.get_data_scaler(config)
        # inverse_scaler = datasets.get_data_inverse_scaler(config)

        # Create a train sampler and loader over the dataset
        data_loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size,
                pin_memory=False, num_workers=2, 
                drop_last=False, shuffle=True
            )


        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        reduce_mean = config.training.reduce_mean
        likelihood_weighting = config.training.likelihood_weighting
        train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                            reduce_mean=reduce_mean, continuous=continuous,
                                            likelihood_weighting=likelihood_weighting)
        eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                            reduce_mean=reduce_mean, continuous=continuous,
                                            likelihood_weighting=likelihood_weighting)

        num_train_steps = config.training.n_iters
        totalSteps = 0

        # In case there are multiple hosts (e.g., TPU pods), only log to host 0
        logging.info("Starting training loop at step %d." % (initial_step,))

        for step in range(initial_step, num_train_steps + 1):
            # Iterate over the dataset
            for step, data in enumerate(data_loader_train):
                k = totalSteps

                loss_model_norm = self.train_step_fn(data, k)

                # Increase the number of steps
                totalSteps += 1




                




                
                loss = loss_model_norm

                # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
                # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
                # batch = batch.permute(0, 3, 1, 2)
                # batch = scaler(batch)
                # Execute one training step
                if totalSteps % config.training.log_freq == 0:
                    logging.info("step: %d, training_loss: %.5e" % (totalSteps, loss.item()))
                    print("step: %d, training_loss: %.5e" % (totalSteps, loss.item()))
                    writer.add_scalar("training_loss", loss, totalSteps)

                # Save a temporary checkpoint to resume training after pre-emption periodically
                if totalSteps != 0 and totalSteps % config.training.snapshot_freq_for_preemption == 0:
                    save_checkpoint(checkpoint_meta_dir, state)

                # Report the loss on an evaluation dataset periodically
                if totalSteps % config.training.eval_freq == 0:
                    with torch.no_grad():
                        # Sample a random batch of eval data
                        random_idx = torch.tensor(np.random.choice(len(self.dataset_eval), config.eval.batch_size, replace=False), dtype=torch.long)
                        eval_batch = self.dataset_eval[random_idx]
                        
                        # Evaluate the model
                        eval_loss = self.eval_fn(eval_batch, k)


                        logging.info("step: %d, eval_loss: %.5e" % (totalSteps, eval_loss.item()))
                        print("step: %d, eval_loss: %.5e" % (totalSteps, eval_loss.item()))
                        writer.add_scalar("eval_loss", eval_loss.item(), totalSteps)

                        del eval_batch
                        del eval_loss

                # Save a checkpoint periodically and generate samples if needed
                if totalSteps != 0 and totalSteps % config.training.snapshot_freq == 0 or totalSteps == num_train_steps:
                    # Save the checkpoint.
                    save_step = totalSteps // config.training.snapshot_freq
                    save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if totalSteps % config.sampling.sample_steps == 0:
                    with torch.no_grad():
                        ema.store(score_model.parameters())
                        ema.copy_to(score_model.parameters())

                        # Sample from the model
                        shape = list(data[0].shape)[1:]
                        sample = self.model_norm.sample(shape, self.sample_N, config.sampling.sample_size)

                        ema.restore(score_model.parameters())
                        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(totalSteps))
                        tf.io.gfile.makedirs(this_sample_dir)
                        nrow = int(np.sqrt(sample.shape[0]))
                        image_grid = make_grid(sample, nrow, padding=2)
                        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        with tf.io.gfile.GFile(
                                os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                            np.save(fout, sample)

                        with tf.io.gfile.GFile(
                                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                            save_image(image_grid, fout)

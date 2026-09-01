import torch
from typing import Callable, Dict, Any
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import logging
logger = logging.getLogger(__name__)


#----------------------------------------------------------------------------------------
def generator_step(g_optimizer: torch.optim.Adam,
                   generator: torch.nn.Module,
                   discriminator: torch.nn.Module,
                   generator_loss: Callable,
                   current_batch_size:int,
                   cfg: Any) -> Dict:
    """
    One training step of the generator.
    Args:
        - g_optimizer: generator optimizer
        - generator: generator model
        - discriminator: discriminator model
        - generator_loss: generator loss calculator
        - current_batch_size: current batch size (last batch may have less size than batch-size)
        - cfg: config contains info on batch_size, latent_dim, and device
    Return:
        generator loss
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        g_optimizer.zero_grad() # 0. init grad to zero for new step

        z = torch.randn(current_batch_size, cfg.latent_dim, 1, 1).to(cfg.device) # fake image N(0,1) no need to be in [-1,1]
        fake_images = generator(z) # pass fake-image through generator

        #pass generator output to discriminator
        D_fake = discriminator(fake_images)  # fake_logits

        #calculate the generator loss
        g_loss = generator_loss(D_fake) # use real loss to flip labels

        # perform backprop, and update weight
        g_loss.backward()
        g_optimizer.step()

        return {'loss': g_loss}
    except Exception as e:
        logger.error(f"Error in get_mean_std: {e}")
        raise

#----------------------------------------------------------------------------------------
def add_instance_noise(images, std_dev):
    """
    Adds Gaussian noise to a batch of images.
    Args:
    - images: input images
    - std_dev: standard deviation (used to create Normal-dist with (0, std_dev)
    Return:
        noisy image if std_dev > 0, otherwise original images
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        if std_dev <= 0:
            return images
        noise = torch.randn_like(images) * std_dev
        return images + noise
    except Exception as e:
        logger.error(f"Error in add_instance_noise: {e}")
        raise

#----------------------------------------------------------------------------------------
def discriminator_step(d_optimizer: torch.optim.Adam,
                       generator: torch.nn.Module,
                       discriminator: torch.nn.Module,
                       discriminator_loss: Callable,
                       real_images: torch.Tensor,
                       current_batch_size: int,
                       cfg: Any,
                       current_sigma: float = 0.0)-> Dict:
    """
    One training step of the generator.
    Args:
        - d_optimizer: discriminator optimizer
        - generator: generator model
        - discriminator: discriminator model
        - discriminator_loss: discriminator loss calculator
        - real_images: real images
        - current_batch_size: current batch size (last batch may have less size than batch-size)
        - cfg: config contains info on batch_size, latent_dim, and device
        - current_sigma of instance noise
    Return:
        generator loss
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        d_optimizer.zero_grad() #init grad to zero for new step


        real_noisy = add_instance_noise(real_images, current_sigma)
        D_real = discriminator(real_noisy) # real_logits

        z = torch.randn(current_batch_size, cfg.latent_dim, 1, 1).to(cfg.device) # fake image N(0,1) no need to be in [-1,1]
        fake_images = generator(z) # train with fake image

        # Compute the discriminator logits/output on fake images
        #D_fake = discriminator(fake_images.detach()) # fake_logits
        fake_noisy = add_instance_noise(fake_images.detach(), current_sigma)
        D_fake = discriminator(fake_noisy) # fake_logits


        # calc total loss,and perform back-propagate, and update weight
        d_loss = discriminator_loss(D_real, D_fake)

        d_loss.backward()
        d_optimizer.step()

        return {'loss': d_loss}
    except Exception as e:
        logger.error(f"Error in discriminator_step: {e}")
        raise

#----------------------------------------------------------------------------------------
class EMA():
    """
    Exponential Moving average class used to smooth generator weights
    """
    def __init__(self, model: torch.nn.Module, decay:float):
        """
        Args:
        - model: model under training
        - decay: decay factor of EMA: new-EMA = decay *EMA +(1-decay)W, keep more from EMA with decay 0.999
        """
        # Move the shadow model to the same device as the original
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.params = list(self.model.parameters())
        self.source_params = list(model.parameters())
        self.buffers = list(self.model.buffers())
        self.source_buffers = list(model.buffers())

    def update(self):
        with torch.no_grad():
            # Update Parameters using the EMA formula
            for p, src_p in zip(self.params, self.source_params):
                p.copy_(self.decay * p + (1.0 - self.decay) * src_p)

            # Buffers (like BatchNorm running stats) should be copied directly
            # They are not averaged; they are state-tracked.
            for b, src_b in zip(self.buffers, self.source_buffers):
                b.copy_(src_b)

#----------------------------------------------------------------------------------------
def initialize_ema(ema_model: torch.nn.Module, source_model: torch.nn.Module):
    """initialize ema_model weight with model weight"""
    ema_model.load_state_dict(source_model.state_dict())

#----------------------------------------------------------------------------------------
def denormalize(images):
    """Transform images from [-1.0, 1.0] to [0, 255] and cast them to uint8."""
    return ((images + 1.) / 2. * 255).astype(np.uint8)

#----------------------------------------------------------------------------------------
def display(fixed_latent_vector: torch.Tensor):
    """ helper function to display images during training """
    fig = plt.figure(figsize=(14, 4))
    plot_size = 16
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
        img = fixed_latent_vector[idx, ...].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = denormalize(img)
        ax.imshow(img)
    #plt.show()

#----------------------------------------------------------------------------------------
def run_training(dataloader: torch.utils.data.DataLoader,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 g_optimizer: torch.nn.Module,
                 d_optimizer: torch.optim.Adam,
                 discriminator_loss: Callable,
                 generator_loss,
                 cfg:Any):
    """
    """
    fixed_latent_vector = torch.randn(cfg.num_image_to_display_during_training, cfg.latent_dim, 1, 1).to(cfg.device)

    ema_gen = EMA(generator, decay=cfg.ema_decay) # set EMA model for genertor
    initialize_ema(ema_gen.model, generator)

    samples = []
    losses = []
    current_step = 1
    for epoch in range(cfg.n_epochs):
        for batch_i, (real_images, _) in enumerate(dataloader): #Ignore labels if training an unconditional GAN:
            real_images = real_images.to(cfg.device)
            current_batch_size = real_images.size(0)

            # add noise to images for both stability (avoiding disc to become too confident) and final quality
            # Calculate current sigma based on a decay schedule
            # Decay from 0.1 to 0 at 60% of total steps, last 40% fine-tuning with no noise
            current_sigma = max(0, 0.1 * (1 - current_step / cfg.total_steps_60_pct))
            current_step += 1
            # train discriminator on all batches
            d_loss = discriminator_step(d_optimizer,
                                        generator,
                                        discriminator,
                                        discriminator_loss,
                                        real_images,
                                        current_batch_size,
                                        cfg,
                                        current_sigma)
            if batch_i%cfg.n_critic == 0: # simply every n_critic train generator once (may improve to avoid missing some batch for training generator)
                g_loss = generator_step(g_optimizer,
                                        generator,
                                        discriminator,
                                        generator_loss,
                                        current_batch_size,
                                        cfg)
                ema_gen.update()

            ####################################

            #if batch_i > 0  and batch_i % print_every == 0:
            if batch_i % cfg.print_every == 0:
                # append discriminator loss and generator loss
                d = d_loss['loss'].item()
                g = g_loss['loss'].item()
                losses.append((d, g))
                # print discriminator and generator loss
                #time = str(datetime.now()).split('.')[0]
                #print(f'{time} | Epoch [{epoch+1}/{n_epochs}] | Batch {batch_i}/{len(dataloader)} | d_loss: {d:.4f} | g_loss: {g:.4f}')
                print(f'Epoch [{epoch+1}/{cfg.n_epochs}] | Batch {batch_i}/{cfg.total_batch} | d_loss: {d:.4f} | g_loss: {g:.4f}')

        # display images during training
        #if epoch % display_every == 0:
        # generator.eval()
        # generated_images = generator(fixed_latent_vector)
        # #samples.append(generated_images)
        # display(generated_images)
        # generator.train()
        ema_gen.model.eval() # Ensure the EMA model is in eval mode
        with torch.no_grad():
            generated_images = ema_gen.model(fixed_latent_vector)# Generate images using the EMA shadow weights
            samples.append(generated_images)
            display(generated_images)# Display or save
        generator.train() # Resume training your RAW generator


    #-------------------------------------
    # Save training generator samples
    with open(cfg.generated_samples_path, 'wb') as f:
        pkl.dump(samples, f)

    np.save(cfg.training_losses_path, np.array(losses))

    # Save generator
    # generator.eval()
    # gen_model_scripted = torch.jit.script(generator)
    # gen_model_scripted.save(gen_model_scripted_path)

    # ema_gen.model.eval()
    # ema_scripted = torch.jit.script(ema_gen.model)
    # ema_scripted.save(ema_gen_model_scripted_path)

    ###
import torch
from typing import Tuple
import logging
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def get_optimizers(generator: torch.nn, discriminator: torch.nn)->Tuple[torch.optim.Adam,torch.optim.Adam]:
    """
    This function should return the optimizers of the generator and the discriminator
    Args:
        - generator: pytorch model of generator in GAN
        - discriminator: pytorch model of discriminator/critic in GAN
    Return:
        optimizer for both generator and discriminator
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        lrg = 1e-4
        lrd = 2e-4 ## Two-Timescale Update Rule (TTUR) ratio 2:1
        beta1=0.5
        beta2=0.999 # default value
        # Create optimizers for the discriminator and generator
        g_optimizer = torch.optim.Adam(generator.parameters(), lrg, [beta1, beta2])
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lrd, [beta1, beta2])

        return g_optimizer, d_optimizer
    except Exception as e:
        logger.error(f"Error in get_optimizers: {e}")
        raise

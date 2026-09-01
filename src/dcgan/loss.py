import torch
import logging
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def real_loss(D_out: torch.Tensor, smooth=False)->torch.Tensor:
    """
    calculating real loss in DCGAN
    Args:
        - D_out: model (generator or discriminator) output
        - smooth: enable/disable label smoothing for loss calculation
    Return:
        loss value
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        batch_size = D_out.size(0)
        # label smoothing
        if smooth:
            # smooth, real labels = 0.9
            labels = torch.ones(batch_size)*0.9
        else:
            labels = torch.ones(batch_size) # real labels = 1
        # move labels to GPU if available
        if torch.cuda.is_available():
            labels = labels.cuda()
        # binary cross entropy with logits loss
        criterion = torch.nn.BCEWithLogitsLoss()
        # calculate loss
        loss = criterion(D_out.squeeze(), labels)
        return loss
    except Exception as e:
        logger.error(f"Error in real_loss: {e}")
        raise
#----------------------------------------------------------------------------------------
def fake_loss(D_out:torch.Tensor)->torch.Tensor:
    """
    calculating fake loss in DCGAN
    Args:
        - D_out: model (generator or discriminator) output
    Return:
        loss value
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size) # fake labels = 0
        if torch.cuda.is_available():
            labels = labels.cuda()
        criterion = torch.nn.BCEWithLogitsLoss()
        # calculate loss
        loss = criterion(D_out.squeeze(), labels)
        return loss
    except Exception as e:
        logger.error(f"Error in fake_loss: {e}")
        raise
#----------------------------------------------------------------------------------------
def generator_loss(fake_logits: torch.Tensor)->torch.Tensor:
    """
    Generator loss, takes the fake scores as inputs.
    Args:
        fake_logits: fake scores
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        loss = real_loss(fake_logits) # use real loss to flip labels
        return loss
    except Exception as e:
        logger.error(f"Error in generator_loss: {e}")
        raise
#----------------------------------------------------------------------------------------
def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor)->torch.Tensor:
    """
    Discriminator loss, takes the fake and real logits as inputs.
    Args:
        - real_logits: real score
        - fake_logits: fake score
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        d_real_loss = real_loss(real_logits, smooth= True)
        d_fake_loss = fake_loss(fake_logits)
        loss = d_real_loss + d_fake_loss
        return loss
    except Exception as e:
        logger.error(f"Error in discriminator_loss: {e}")
        raise
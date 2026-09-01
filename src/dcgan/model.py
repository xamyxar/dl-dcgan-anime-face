import logging
import torch
import torch.nn as nn
from typing import Any, Optional

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """A convolutional block: Conv -> BatchNorm (optional) -> ReLU/LeakyReLU.
    Args:
        - in_channels: Number of input channels.
        - out_channels: Number of output feature maps.
        - cfg: Configuration object containing other parameters (e.g. batch_norm (apply or not) or negative_slope of LeakyReLU (zero means ReLU)
        - local_batch_norm: overwrite global batchnorm setting mainly for first layer
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Any, local_batch_norm:Optional[bool]=None):
        super().__init__()
        if local_batch_norm is None:
            self.batch_norm = cfg.batch_norm
        else:
            self.batch_norm = local_batch_norm # overwrite global setting for this specific layer (in DCGAN no batch for first layer)

        if cfg.bias_negate_batch_norm and self.batch_norm:
            bias_ = False
        else:
            bias_ = True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride=2, padding=1, bias = bias_)
        #self.batch_norm = cfg.batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if cfg.negative_slope >0:
            self.activation = nn.LeakyReLU(cfg.negative_slope) # default negative_slope = 1e-2
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x

#----------------------------------------------------------------------------------------
class DeconvBlock(nn.Module):
    """A transposed convolutional block: ConvTranspose -> BatchNorm (optional) -> ReLU/LeakyReLU.
    Args:
        - in_channels: Number of input channels.
        - out_channels: Number of output feature maps.
        - cfg: Configuration object containing other parameters (e.g. batch_norm (apply or not) or negative_slope of LeakyReLU (zero means ReLU)
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Any, local_kernel_size:Optional[int]=None):
        super().__init__()
        if local_kernel_size is None:
            kernel_size_ = 4
            stride_ = 2
            padding_ = 1
        else: # special first layer case
            kernel_size_ = local_kernel_size
            stride_ = 1
            padding_ = 0

        # when bias is not given it set to True, [TODO] We may set to False if Conv2d follow by a BatchNorm2d
        if cfg.bias_negate_batch_norm and cfg.batch_norm:
            bias_ = False
        else:
            bias_ = True
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size_, stride = stride_, padding =padding_, bias = bias_)
        self.batch_norm = cfg.batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if cfg.negative_slope >0:
            self.activation = nn.LeakyReLU(cfg.negative_slope) # default negative_slope = 1e-2
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x

#----------------------------------------------------------------------------------------
class Discriminator(nn.Module):
    """Convolutional Discriminator (Critic) for DCGAN.
    Args:
        - cfg: Configuration object containing required parameter:
            conv_dim: Base feature channels count for initial layer (default: 64).
            num_conv_layer: Total number of downsampling conv layers (default: 4).
            image_size: Input spatial height/width (default: 64).
    """
    def __init__(self, cfg:Any):
        super().__init__()

        # current values
        in_channel = 3  # RGB image input
        out_channel = cfg.conv_dim # number of filters in initial layers
        spatial_dim = cfg.image_size
        batch_norm_ = False # No batch Norm in first layer

        conv_layer_list = []
        for _ in range(cfg.num_conv_layer):
            conv_layer_list.append(ConvBlock(in_channel, out_channel, cfg, batch_norm_))
            # Each ConvBlock with stride=2 halves the dimension
            spatial_dim = spatial_dim // 2
            in_channel = out_channel
            out_channel *= 2
            batch_norm_ = None # Use  global settings from cfg.batch_norm for all other layer

        self.conv_layers = nn.Sequential(*conv_layer_list)

        # Flattened dimension after spatial reduction
        # assume we start with RGB-3-channel 64x64 image, and we have 4 layers
        # I: 3, 64x64--> O1: 64, 32x32, O2: 128, 16x16, O3: 256, 8x8, O4: 512, 4x4
        # at the end of loop we have in_channel CH with spatial_dim
        #self.final_channels = in_channel
        #self.spatial_dim = spatial_dim
        #self.flatten_dim = self.final_channels * self.spatial_dim * self.spatial_dim

        # Instead of Flatten + FC, use a current_dim x current_dim Convolution kernel
        # This will keep 4D dimension that expected in GAN architecture
        # otherwise if we use Flatten and FC we end up 2D size [Batch,1] then
        # we have to add extra x.view(-1, 1, 1, 1), which doesn't sound good!
        # Extra Note: The Conv2d trick is not a localized convolution in this specific instance—because the kernel spans the entire feature map,
        # it acts as a fully connected layer in disguise.
        # It gives you the exact same weights and mathematical output, but keeps your tensor dimensions clean.

        self.final_conv = torch.nn.Conv2d(in_channel, 1, kernel_size=spatial_dim, stride=1, padding=0) # actuall kernel_size =4  for above example
        #self.sigmoid = nn.Sigmoid() # No-sigmoid when we use BCEWithLogitsLoss for loss which combine BCE and sigmoid for more stable calc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(x)
        x = self.final_conv(x)   # Shape: [Batch, 1, 1, 1] <--- 4D as expected by GAN
        #x = self.sigmoid(x) No-sigmoid when we use BCEWithLogitsLoss for loss which combine BCE and sigmoid for more stable calc
        return x


class Generator(nn.Module):
    """Deconvolutional Generator (Decoder latent to image)) for DCGAN. (Image Generator from latent)
    Maps latent representation z to image space.
    Args:
        - cfg: Configuration object containing required parameter:
            latent_dim: Dimension of latent Gaussian distribution z.
            conv_dim: Base feature channels count for initial layer (default: 64).
            num_conv_layer: Total number of downsampling conv layers (default: 4).
            image_size: Input spatial height/width (default: 64).
            last_activation_sigmoid: if True--> sigmoid, we use Sigmoid when image [0,1] and Tanh when image [-1,1] (DCGAN we use Tanh)
    """
    def __init__(self, cfg:Any):
        super().__init__()
        self.spatial_dim = cfg.image_size // (2**cfg.num_conv_layer) # initial spatial dim = last spatioal dim in Discriminator
        self.init_channels = (2 ** (cfg.num_conv_layer - 1)) * cfg.conv_dim # initial out-channel


        # self.flatten_dim = self.init_channels * self.spatial_dim * self.spatial_dim
        # # Project 1D latent z to 4D initial feature volume
        # self.fc_decoder = nn.Linear(cfg.latent_dim, self.flatten_dim)
        # in_channel = self.init_channels
        # out_channel = in_channel // 2

        in_channel = cfg.latent_dim
        out_channel = self.init_channels
        local_kernel_size = self.spatial_dim #


        deconv_layer_list = []

        for j in range(cfg.num_conv_layer):
            deconv_layer_list.append(DeconvBlock(in_channel, out_channel, cfg, local_kernel_size))
            in_channel = out_channel
            out_channel = out_channel // 2
            local_kernel_size = None # for all other layers used default settings

        # Final reconstruction deconv block (upsamples to output image dimensions) no ReLU or BatchNorm, only ConvTranspose2d layer
        deconv_layer_list.append(nn.ConvTranspose2d(in_channel, 3, kernel_size=4, stride=2, padding=1))

        self.deconv_layers = nn.Sequential(*deconv_layer_list)
        # Sigmoid for pixel intensities normalized in range [0, 1], [-1,1] if normalized to [-1,1]
        if cfg.last_activation_sigmoid:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.fc_decoder(z)
        # # Reshape to 4D tensor: [Batch, Channels, Height, Width]
        # x = x.view(-1, self.init_channels, self.spatial_dim, self.spatial_dim)

        # we include the above inside deconv-layer, with similar trick used in Discriminator
        x = self.deconv_layers(x)
        x = self.last_activation(x)
        return x
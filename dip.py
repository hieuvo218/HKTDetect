"""
This class is used to perform digital image processing techniques
"""

import numpy as np
import math
from deep_learning import *


def box_blur(x, kernel_size=3, padding_mode='zeros', channel_last: bool = True):
    if channel_last: 
        in_channels = x.shape[-1]
    else:
        in_channels = x.shape[1]

    stride = 1
    padding = kernel_size // 2       
    conv_x = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)

    weight = 1 / (kernel_size * kernel_size) * np.ones((out_channels, in_channels, kernel_size, kernel_size)) # normalize with kernel size
    conv_x.set_weight(weight)

    return conv_x(x)


def gaussian(x, sigma=7):
    return math.exp(-x**2/(2*sigma**2))


def gaussian_blur(x, kernel_size=3, sigma=1, padding_mode='zeros', channel_last: bool = True):
    if channel_last:
        in_channels = x.shape[-1]
    else:
        in_channels = x.shape[1]
    out_channels = in_channels  # just blur image, therefore preserve channels

    stride = 1
    padding = kernel_size // 2
    conv_x = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)

    # Initialize weights with gaussian kernel
    weights = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
    x_c, y_c = kernel_size // 2, kernel_size // 2   # center point
    sum = 0

    for y in range(kernel_size):
        for x in range(kernel_size):
            weights[0, :, y, x] = gaussian(x - x_c) * gaussian(y - y_c)
            sum += weights[0, 0, y, x]

    weights = weights / sum
    conv_x.set_weight(weights)

    return conv_x(x)
import numpy as np
import math


class Module:
    def __init__(self):
        self.weight = None
    
    def forward(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self, weight):
        self.weight = weight


class Conv1D(Module):
    """1D Convolutional Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        if padding_mode != 'zeros':
            raise ValueError("Only 'zeros' padding mode is supported.")
        else:
            self.padding_value = 0
        # Initialize weights and biases
        self.weight = np.random.randn(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = np.random.randn(out_channels)
        else:
            self.bias = None
    
    def forward(self, x):
        emb_len = x.shape[-1]
        x = np.pad(x, ((0,0), (self.padding, self.padding)), 'constant', constant_values=self.padding_value)
        output_len = math.floor((emb_len + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        kernel_mat = np.zeros((self.in_channels, emb_len + 2*self.padding, output_len))
        st = 0
        l = []
        for c in range(self.out_channels):
            st = 0
            for i in range(output_len):
                kernel_mat[:, st:st+self.kernel_size, i] = self.weight[c, :, :]
                st += self.stride
            conv_mat = np.sum(x[:,None,:] @ kernel_mat, axis=0)
            l.append(conv_mat)
        conv_mat = np.concatenate(l, axis=0)
        return conv_mat
    

class Conv2D(Module):
    """2D Convolutional Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_y, self.kernel_x = kernel_size, kernel_size
        elif isinstance(kernel_size, tuple):
            self.kernel_y, self.kernel_x = kernel_size

        if isinstance(padding, int):
            self.padding_y, self.padding_x = padding, padding
        elif isinstance(padding, tuple):
            self.padding_y, self.padding_x = padding

        if isinstance(stride, int):
            self.stride_y, self.stride_x = stride, stride
        elif isinstance(stride, tuple):
            self.stride_y, self.stride_x = stride

        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        if padding_mode != 'zeros':
            raise ValueError("Only 'zeros' padding mode is supported.")
        else:
            self.padding_value = 0
        # Initialize weights and biases
        self.weight = np.random.randn(out_channels, in_channels, self.kernel_y, self.kernel_x)
        if bias:
            self.bias = np.random.randn(out_channels)
        else:
            self.bias = None


    def forward(self, x):
        H, W, C = x.shape
        x = np.transpose(x, (2, 0, 1))
        x = np.pad(x, ((0, 0), (self.padding_y, self.padding_y), (self.padding_x, self.padding_x)), 'constant', constant_values=self.padding_value)
        H_out = math.floor((H + 2 * self.padding_y - self.dilation * (self.kernel_y - 1) - 1) / self.stride_y + 1)
        W_out = math.floor((W + 2 * self.padding_x - self.dilation * (self.kernel_x - 1) - 1) / self.stride_x + 1)

        conv_mat = np.empty((self.out_channels, H_out, W_out))
        st_y, st_x = 0, 0
        for i in range(H_out):
            st_x = 0
            for j in range(W_out):
                for c in range(self.out_channels):
                    conv_output = x[:, st_y : st_y + self.kernel_y, st_x : st_x + self.kernel_x] * self.weight[c, :, :, :]
                    conv_mat[c, i, j] = np.sum(conv_output)
                st_x += self.stride_x
            st_y += self.stride_y
        return conv_mat
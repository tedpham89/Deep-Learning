"""
2d Convolution Module. 

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None

        # Collect dimension value
        N, _, H , W = x.shape
        # From the lecture, the feature map size is:
        # H_fm = (H + padding - kernels)//stride + 1. We also pad on both side.
        # Calculate feature map 2d size
        H_fm = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_fm = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        # np.pad(x,pad_width = self.padding, mode='constant',constant_values=0)
        # Fix original method pad on all 4 dimensions
        # When padding , we only pad on 2D dimension H&W, not on N and C dimension.
        x_padded = np.pad(x,
                          pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                          mode='constant',
                          constant_values=0
                          )

        # Init output
        out = np.zeros((N,self.out_channels,H_fm,W_fm))

        # Forward Pass
        for n in range(N): #batch
            for f in range(self.out_channels): #filter
                for w in range(W_fm): # we go top to bottom first before going right
                    for h in range(H_fm):
                        # corresponding sub input
                        top_h = h * self.stride
                        left_w = w * self.stride
                        bot_h = top_h + self.kernel_size
                        right_w = left_w + self.kernel_size

                        # print("weight.shape",self.weight.shape)
                        # print("bias shape", self.bias.shape)
                        # print("x_window", x_padded[n,:,top_h:bot_h,left_w:right_w].shape)
                        # print("out",out.shape)
                        # Multiply piece wise and sum acrros in_channel
                        out[n,f,h,w] = np.sum(x_padded[n,:,top_h:bot_h,left_w:right_w] * self.weight[f]) + self.bias[f]

        # store x to cache for back prop
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        # print("dout.shape", dout.shape)
        # # dout(N,self.out_channels, H',W')
        # print("x_shape",x.shape)
        # print("weight.shape",self.weight.shape)
        # print("bias shape", self.bias.shape)

        # collect dimension value, similar to forward section
        N, C, H, W = x.shape
        H_fm = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_fm = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        x_padded = np.pad(x,
                          pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                          mode='constant',
                          constant_values=0
                          )
        
        # Init gradients
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)

        # Based on the lecture, the I take advantage of the fact that backward pass 
        # for dx is the convolution of the 180o kernel flip and dout
        # Resource:
        # https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
        # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf

        # First dilate and padded the upstream so convolution output produce same dimension as dx
        dilation_factor = self.stride - 1
        H_dilate = H_fm + (H_fm - 1) * dilation_factor
        W_dilate = W_fm+ (W_fm - 1) * dilation_factor

        # Dilate and padding dout
        dout_dilated = np.zeros((N,self.out_channels,H_dilate,W_dilate),dtype=dout.dtype)
        dout_dilated[:,:,::self.stride,::self.stride] = dout

        pad_dout = self.kernel_size - 1
        dout_padded = np.pad(dout_dilated,((0,0),(0,0),(pad_dout,pad_dout),(pad_dout,pad_dout)), mode='constant', constant_values=0)

        # Flip the kernel/filter 180o
        w_flipped = np.flip(self.weight, axis = (2,3))

        # w_flipped = np.rot90(self.weight, k = 2, axes=(2,3))
        # Backward Pass
        for n in range(N): #batch
            for f in range(self.out_channels): #filter
                for w in range(W_fm): # we go top to bottom first before going right
                    for h in range(H_fm):
                        # corresponding sub input
                        top_h = h * self.stride
                        left_w = w * self.stride
                        bot_h = top_h + self.kernel_size
                        right_w = left_w + self.kernel_size

                        # print("weight.shape",self.weight.shape)
                        # print("bias shape", self.bias.shape)
                        # print("x_window", x_padded[n,:,top_h:bot_h,left_w:right_w].shape)
                        # print("out",out.shape)
                        # Gradients wrt weights
                        dw[f] += x_padded[n,:,top_h:bot_h,left_w:right_w] * dout[n,f,h,w]
                # sum dout for gradient wrt bias
                db[f] += np.sum(dout[n,f])
        # This second part is normal convolution forward pass of new dout and the kernel flip
        for n in range(N):
            for c in range(C):
                for w in range(dx_padded.shape[3]): # we go top to bottom first before going right
                    for h in range(dx_padded.shape[2]):

                        for f in range(self.out_channels):
                            top_h = h 
                            left_w = w 
                            bot_h = top_h + self.kernel_size
                            right_w = left_w + self.kernel_size
                            dx_padded[n,c,h,w] += np.sum(dout_padded[n,f,top_h:bot_h,left_w:right_w] * w_flipped[f,c,:,:])


        dx = dx_padded[:,:,self.padding:H+self.padding,self.padding:W+self.padding]

        self.dx,self.dw,self.db = dx, dw, db









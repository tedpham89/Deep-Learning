"""
2d Max Pooling Module. 

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        N, C, H, W = x.shape
        # print("N:",N,"C:",C,"H:",H,"W:",W)
        # formula of output size is (input+padding-kennel)//stride + 1
        # the output map is size (N,C,H_new, W_new)
        # Calculate output map
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1

        out = np.zeros((N,C,H_out,W_out))

        # starting from top left corner as (0,0) we slide the kernel/filter along row then column
        # graping highest x value
        # Iterate through each sample N and channel C
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        top_left_w = w * self.stride
                        top_left_h = h * self.stride
                        top_right_w = top_left_w + self.kernel_size
                        bot_left_h = top_left_h + self.kernel_size

                        # each filter is a square and grap the highest value within that filter window
                        filter_region = x[n,c,top_left_h:bot_left_h,top_left_w:top_right_w]
                        # max operation, fill up the output map
                        out[n,c,h,w] = np.max(filter_region)
                
        #store cache for backprop
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        N,C,H,W = x.shape
        self.dx = np.zeros(x.shape, dtype = np.float64)
        # print("N:",N,"C:",C,"H:",H,"W:",W)
        # print("H_out:",H_out,"W_out:",W_out)
        
        # Similar to foward, we move the kernel and find the position of the max values 
        # and update it with gradient of that specific position x
        # Loop similar to forward step
        for n in range(N):
            for c in range(C):
                # for h in range(H_out):
                #     for w in range(W_out):
                # originally we can go left first then down and left
                # but we can also go down first then go left.
                for w in range(W_out):
                    for h in range(H_out):
                        top_left_w = w * self.stride
                        top_left_h = h * self.stride
                        top_right_w = top_left_w + self.kernel_size
                        bot_left_h = top_left_h + self.kernel_size

                        # filter region:
                        filter_region = x[n,c,top_left_h:bot_left_h,top_left_w:top_right_w]
                        # The tips about using np.unravel_index help locating the position of before flaten
                        # it will return 2d position for h and w or row and column
                        position = np.unravel_index(np.argmax(filter_region), filter_region.shape)
                        # use += because sometime the pooling is overlap, happen when kernel > stride

                        self.dx[n,c,top_left_h + position[0], top_left_w +position[1]] += dout[n,c,h,w]

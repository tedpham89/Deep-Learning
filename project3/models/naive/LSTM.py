"""
LSTM model.  

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Follow exact implementation on https://static.us.edusercontent.com/files/Xd1TcLyqvYoc6WmVd8yGQV3u

        # i_t: input gate
        # X shape (N, input_size) ,  weights (input_size, hidden_size) = (N,hidden_size)
        # bias (hidden_size)
        # h (N, hidden_size) , weight (hidden_size, hidden_size) = (N,hidden_size), so bias (hidden_size)
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t: the forget gate
        # X shape (N, input_size) ,  weights (input_size, hidden_size) = (N,hidden_size)
        # bias (hidden_size)
        # h (N, hidden_size) , weight (hidden_size, hidden_size) = (N,hidden_size), so bias (hidden_size)
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t: the cell gate
        # X shape (N, input_size) ,  weights (input_size, hidden_size) = (N,hidden_size)
        # bias (hidden_size)
        # h (N, hidden_size) , weight (hidden_size, hidden_size) = (N,hidden_size), so bias (hidden_size)
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t: the output gate
        # X shape (N, input_size) ,  weights (input_size, hidden_size) = (N,hidden_size)
        # bias (hidden_size)
        # h (N, hidden_size) , weight (hidden_size, hidden_size) = (N,hidden_size), so bias (hidden_size)
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""


        N, seq_len, input_size = x.size()

        # Initialize hidden and cell states to zero, shape (N, hidden_size)
        h_t = torch.zeros(N, self.hidden_size, dtype=x.dtype)
        c_t = torch.zeros(N, self.hidden_size, dtype=x.dtype)
        # h_t sometime called short term memory, and c_t short term memory

        for n in range(seq_len):
            x_t = x[:, n, :]  # shape (batch_size, input_size)

            # i_t: input gate - use sigmoid activation function, % potential of new input to remember
            i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h_t @ self.W_hi + self.b_hi)

            # f_t: forget gate - use sigmoid activation function, % long term to remember
            f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h_t @ self.W_hf + self.b_hf)

            # g_t: cell gate - use tanh function, potential new input to remember
            g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h_t @ self.W_hg + self.b_hg)

            # o_t: output gate - use sigmoid, % of h_t to remember for new output h_t
            o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h_t @ self.W_ho + self.b_ho)

            # Next cell state
            # Next cell state, retain part of old state + new potential mem/input
            c_t = f_t * c_t + i_t * g_t

            # Next hidden state, use formula from class instruction. Note this is new output or new hidden state.
            h_t = o_t * torch.tanh(c_t)


        return (h_t, c_t)

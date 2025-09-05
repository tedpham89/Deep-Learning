"""
S2S Encoder model.  

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type

        # Follow instruction initialize steps
        # Init embedding layer using nn.embedding
        self.embedding_layer = nn.Embedding(input_size, emb_size)

        # Recurrent layer based on model_type, instruction say batch first = true
        if model_type == "RNN":
            self.RNNLayer = nn.RNN(input_size=emb_size, hidden_size=encoder_hidden_size, batch_first=True)
        else:
            self.RNNLayer = nn.LSTM(input_size=emb_size, hidden_size=encoder_hidden_size, batch_first=True)

        # Linear layer, the instruction say linear to ReLU to Linear to tanh
        # first linear will take encoder hidden size - encoder hidden size output
        # last layer must have output decoder hidden size, ReLU and tanh wont change dimension
        self.fc1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.fc2_activation = nn.Tanh()

        # Init drop out layer
        self.dropout_layer = nn.Dropout(dropout)



    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """



        # Follow exactly what has been laying out by instruction:
        # embedding - dropout - RNN layer - FC linear1 - ReLU - FC linear 2- Tanh - output
        # print("input shape", input.shape)
        input_embedded = self.embedding_layer(input)
        # Dropout
        input_embedded = self.dropout_layer(input_embedded)
        # Recurrent layer RNN or LSTM
        output, hidden = self.RNNLayer(input_embedded)
        # the hidden in LSTM is a tuple (hidden,cell), here we use the name hidden only

        # Follow instruction, we do not let cell state LSTM go through FC layer
        if self.model_type == "RNN":
            hidden = self.fc1(hidden)
            hidden = self.fc1_activation(hidden)
            hidden = self.fc2(hidden)
            hidden = self.fc2_activation(hidden)
        else:
            # Unpack hidden and cell state of LSTM
            hidden_state, cell_state = hidden
            # Applying FC layer
            hidden_state = self.fc1(hidden_state)
            hidden_state = self.fc1_activation(hidden_state)
            hidden_state = self.fc2(hidden_state)
            hidden_state = self.fc2_activation(hidden_state)

            #Repacking
            hidden = (hidden_state, cell_state)

        # print("output.shape", output.shape, "hidden shape", hidden.shape)        
        return output, hidden

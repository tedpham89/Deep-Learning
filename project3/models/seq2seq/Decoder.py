"""
S2S Decoder model.  

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention


        # Follow instruction of starting code
        # Init Embedding layer
        self.embedding_layer = nn.Embedding(self.output_size, self.emb_size)

        # Recurrent layer
        if model_type == "RNN":
            self.RNNLayer = nn.RNN(input_size=emb_size,hidden_size=encoder_hidden_size, batch_first=True)
        else:
            self.RNNLayer = nn.LSTM(input_size=emb_size,hidden_size=encoder_hidden_size, batch_first=True)

        # FC Layer with logsoftmax
        self.fc_output_layer = nn.Linear(decoder_hidden_size, output_size)
        self.fc_output_activation = nn.LogSoftmax(dim = 1)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Attention is True, Linear layer to downsize, because concatenate context of hidden size and input emb_size
        # downsize to input-emb_size for the RNN layer.
        if self.attention == True:
            self.fc_attention_downsize = nn.Linear(self.encoder_hidden_size + self.emb_size, self.emb_size)


    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        # The hidden came from decoder have shape (1,N,dec_hidden_size) 
        # Output from encoder has shape (N, seq_size, hidden_size)
        # querry need to have shape (N, 1, hidden_size)
        # print("hidden shape",hidden.shape)
        # print("encode output shape", encoder_outputs.shape)
        hidden = hidden.squeeze(0)  # (N,hidden_size)
        hidden = hidden.unsqueeze(1) # (N,1,hidden_size)

        similarity = nn.functional.cosine_similarity(hidden,encoder_outputs, dim = 2) # dot product along hidden_size
        # print("similarity shape",similarity.shape)
        

        # use softmax to convert to probability (N,T)
        attention_prob = nn.functional.softmax(similarity, dim = 1) # accross sequece size or T 
        # print(attention_prob)

        # Unsqueeze to size (N,1,T) as instruction
        attention_prob = attention_prob.unsqueeze(1)

        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """


        # Apply dropout it embedding input
        input_embedded = self.embedding_layer(input)  #(N, 1, emb_size)
        input_embedded = self.dropout_layer(input_embedded)
        # print("input embed shape", input_embedded.shape)

        # Since attention only apply to hidden state in LSTM and hidden from LSTM return a tuple
        # Need to seperate cell and hidden state
        if self.model_type == "RNN":
            hidden_state = hidden
            querry = hidden
        else:
            # Unpack hidden and cell state
            hidden_state, cell_state = hidden
            querry = hidden_state
        # print("hidden_state shape", hidden_state.shape)  
        # hidden_state shape (1,N,decoder_hidden)

        if self.attention == True and encoder_outputs is not None:
            attention_probs = self.compute_attention(querry, encoder_outputs) # (N,1,T)

            # Encoder_output shape (N,T,encoder_hidden_size)
            # At each batch, we want weighted sum of (1,T) * (T,encoder_hidden_s) = (1,encoder_hidden)
            weighted_sum = torch.bmm(attention_probs, encoder_outputs) # (N,1,encoder_hidden)
            # print("weighted sum shape",weighted_sum.shape)

            # weight sum (N,1,encoder-hidden) , embedded input (N,1,emb_size)
            # concatenate (N,1,encoder_hiddensize+emb_size)
            context = torch.cat((weighted_sum,input_embedded), dim = 2)

            # Downsize to match (N, 1, emb_size) shape
            RNN_input = self.fc_attention_downsize(context) 
        else:
            RNN_input = input_embedded

        # Feed hidden state, and input to RNN model
        RNN_output, hidden = self.RNNLayer(RNN_input, hidden)

        

        # print("RNN_output", RNN_output.shape)
        # Reshape output, and feed through output layer logsoftmax
        RNN_output = RNN_output.squeeze(1) # shape (N, decoder_hidden)
        output = self.fc_output_layer(RNN_output)
        output = self.fc_output_activation(output) # shape (N, output_size)
        
        return output, hidden

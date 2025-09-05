import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        
        self.decoder = decoder.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #


    def forward(self, source):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]


        # Get the last hidden representation from the encoder. Use it as
        # the first hidden state of the decoder   
        encoder_output, hidden = self.encoder.forward(source)

        
        # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, decoder_output_size)
        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size, device = self.device)

        # Feed first input to the decoder, this is <SOS>
        # Input feed to decoder shape is (N,1)
        decoder_input = source[:,0] # shape(N)
        decoder_input = decoder_input.unsqueeze(1) #shape(N,1)

        for t in range(seq_len):
            decoder_output, hidden = self.decoder.forward(decoder_input, hidden, encoder_output)

            # Save result
            outputs[:,t,:] = decoder_output

            # Prediction
            decoder_output = decoder_output.argmax(dim = 1) # shape (N)
            # print("decoder out",decoder_output.shape)

            # Update input for next sequence
            decoder_input = decoder_output.unsqueeze(1) # shape (N,1)



        return outputs

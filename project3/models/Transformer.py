"""
Transformer model.  

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        

        self.embeddingL = nn.Embedding(num_embeddings=self.input_size,
                                        embedding_dim=self.hidden_dim)      #initialize word embedding layer
        self.posembeddingL = nn.Embedding(num_embeddings=self.max_length,
                                        embedding_dim=self.hidden_dim)   #initialize positional embedding layer
        
        # Problem when running 4.6 , need to move to same device
        self.embeddingL = self.embeddingL.to(self.device)
        self.posembeddingL = self.posembeddingL.to(self.device)


        # Deliverable 2: Initializations for multi-head self-attention.              #
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)


        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Init 2 FCLayer and ReLU
        self.FeedForwardLayer1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.FeedForwardLayer1_activation = nn.ReLU()
        self.FeedForwardLayer2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        # Add Norm
        self.FeedForwardLayer_norm = nn.LayerNorm(self.hidden_dim)


        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        # Init last layer, change dimension from hidden to output_size
        self.Last_Layer = nn.Linear(self.hidden_dim, self.output_size)

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
        # print("D5 inputs", inputs.shape)
        # shape (N,T)
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #

        # Embeding
        inputs = self.embed(inputs)  # (N,T,H)    

        # Multi Head self attention
        outputs = self.multi_head_attention(inputs)  #(N,T,H)

        # Feed forward, already handle shape
        outputs = self.feedforward_layer(outputs)

        # Final Layer
        outputs = self.final_layer(outputs)
        # print("D5 output shape", outputs.shape)

        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """

        # Unpack
        N, T = inputs.shape

        # input shape (N,T)
        token_embed = self.embeddingL(inputs) # shape (N,T,hidden)

        # position_embed need to have same shape (N,T,hidden_dim)
        # Because we have max_length, first split the length in equal position for all word in squence
        length_intevals = torch.linspace(0, self.max_length - 1, steps=T, device=inputs.device).long() # Shape T
        length_intevals = length_intevals.unsqueeze(0).expand(N, -1)  # Shape: (N, T)

        # Pass through embedding layer
        position_embed = self.posembeddingL(length_intevals)

        # Combine word and positional embeddings
        embeddings = token_embed + position_embed
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """

        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Unpack for shape_length
        N, T, H = inputs.shape  # Batch size (N), sequence length (T), hidden dim (H)
    
        # Calculate the Query, Keys and Values of first Head:
        # The FC layer change hidden_dim to different dimension
        K1 = self.k1(inputs) # (N,T,dim_k)
        V1 = self.v1(inputs) # (N,T,dim_v)
        Q1 = self.q1(inputs) # (N,T,dim_q)

        # Similarly, calculate QVK for second head
        K2 = self.k2(inputs) # (N,T,dim_k)
        V2 = self.v2(inputs) # (N,T,dim_v)
        Q2 = self.q2(inputs) # (N,T,dim_q)

        # Calculate self attention for head 1
        # attention1 = Q1 @ K1.transpose(-2,-1) / np.sqrt(self.dim_k)
        attention1 = (Q1 @ K1.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.dim_k, dtype = torch.float32)) # (N,T,T)
        attention1 = self.softmax(attention1) # (N,T,T)
        # given first dim is preserve, (N,T,T) @ (N,T,dimV) 
        attention1 = attention1@V1 # shape (N,T,dim_v)
        # print("attention1",attention1.shape)

        # Calculate self attention for head 2
        attention2 = (Q2 @ K2.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.dim_k, dtype = torch.float32)) # (N,T,T)
        attention2 = self.softmax(attention2) # (N,T,T)
        attention2 = attention2@V2 # shape (N,T,dim_v)
        # print("attention2",attention2.shape)

        # Concatenate attention
        attention_concat = torch.cat((attention1, attention2), dim = 2)  # along dim_v , shape (N,T, dim_v * 2)
        # Head projection and add norm
        outputs = self.attention_head_projection(attention_concat) # Shape change to (N,T,hidden_dim)
        outputs = self.norm_mh(inputs + outputs)  # same shape


        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        # print("FFL input shape", inputs.shape) (N,T,hidden_dim)
        outputs = self.FeedForwardLayer1(inputs) # (N,T,dim_FF)
        outputs = self.FeedForwardLayer1_activation(outputs)
        outputs = self.FeedForwardLayer2(outputs)   # shape (N,T,hidden_dim)

        # Add Norm
        outputs = self.FeedForwardLayer_norm(inputs + outputs)

        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        # Apply last fully connected layer, convert to output dimension
        outputs = self.Last_Layer(inputs) #shape (N,T,V)

        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)


        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #

        self.transformer = nn.Transformer( d_model=self.hidden_dim,
                                          nhead=self.num_heads,
                                          num_decoder_layers=num_layers_dec,
                                          num_encoder_layers=num_layers_enc,
                                          dim_feedforward=self.dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True) # input tensor shape (N,T,hidden_dim)


        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #

        # Do not change the order for these variables
        self.srcembeddingL = nn.Embedding(num_embeddings=self.input_size,
                                          embedding_dim=self.hidden_dim,
                                          padding_idx=self.pad_idx)      #embedding for src
        self.tgtembeddingL = nn.Embedding(num_embeddings=self.output_size,
                                          embedding_dim=self.hidden_dim,
                                          padding_idx=self.pad_idx)       #embedding for target
        self.srcposembeddingL = nn.Embedding(num_embeddings=self.max_length,
                                             embedding_dim=self.hidden_dim)    #embedding for src positional encoding
        self.tgtposembeddingL = nn.Embedding(num_embeddings=self.max_length,
                                             embedding_dim=self.hidden_dim)     #embedding for target positional encoding

        self.srcembeddingL = self.srcembeddingL.to(self.device)
        self.tgtembeddingL = self.tgtembeddingL.to(self.device)
        self.srcposembeddingL = self.srcposembeddingL.to(self.device)
        self.tgtposembeddingL = self.tgtposembeddingL.to(self.device)




        # Deliverable 3: Initialize what you need for the final layer.
        self.Last_Layer = nn.Linear(self.hidden_dim, self.output_size)


    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """

        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        # print("src shape",src.shape)
        outputs = None
        # print("tgt shape",tgt.shape)
        # print(tgt)
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)
        # print("tgt shape",tgt.shape)
        # print(tgt)
        # Unpack
        N, T_src = src.shape
        T_tgt = tgt.shape[1]
        # if N != tgt.shape[0] and T != tgt.shape[1]:
        #     return ValueError("expect src and tgt same shape")

        # We just need to follow all the step layout here
        # embed src and tgt for processing by transformer
        src_token_embed = self.srcembeddingL(src)    # shape (N,T_src,hidden)
        tgt_token_embed = self.tgtembeddingL(tgt)    # shape (N,T_src,hidden)

        # position embed
        length_intevals_src = torch.linspace(0, self.max_length - 1, steps = T_src, device=self.device).long().to(self.device) #Shape T
        length_intevals_src = length_intevals_src.unsqueeze(0).expand(N, -1) #Shape (N,T)
        length_intevals_tgt = torch.linspace(0, self.max_length - 1, steps = T_tgt, device=self.device).long().to(self.device) #Shape T
        length_intevals_tgt = length_intevals_tgt.unsqueeze(0).expand(N, -1) #Shape (N,T)
        
        # Pass through embedding layer
        src_position_embed = self.srcposembeddingL(length_intevals_src)
        tgt_position_embed = self.tgtposembeddingL(length_intevals_tgt)
        
        # Combine word and positional embeddings
        src_embed = src_token_embed + src_position_embed
        tgt_embed = tgt_token_embed + tgt_position_embed
        # print("src embed",src_embed.shape)

        # create target mask and target key padding mask for decoder - Both have boolean values
        tgt_mask = torch.triu(torch.ones(T_tgt, T_tgt), diagonal=1).bool().to(self.device) #shape (T_tgt, T_tgt)
        # print("tgt_mask",tgt_mask.shape)
        tgt_key_mask = (tgt == self.pad_idx)  # (N,T_tgt)

        # invoke transformer to generate output
        outputs = self.transformer(src = src_embed,
                                   tgt = tgt_embed,                                   
                                   tgt_mask = tgt_mask,
                                   tgt_key_padding_mask=tgt_key_mask)

        # pass through final layer to generate outputs
        outputs = self.Last_Layer(outputs)
        # print(outputs.shape)
        # print(outputs[:,0,0])
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        
        N,T = src.shape
        # src = src.to(device = self.device)

        # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, output_size)
        outputs = torch.zeros((N,T,self.output_size),dtype = torch.float, device = self.device)

        # initially set tgt as a tensor of <pad> tokens with dimensions (batch_size, seq_len)
        tgt  = torch.full((N,T),dtype = torch.long, fill_value= self.pad_idx, device = self.device)
        # tgt = self.add_start_token(tgt)
        tgt[:,0] = src [:,0]

        # print("src", src.shape,src)
        # print("tgt", tgt.shape,tgt)
        for t in range(T):
            # print("Time:",t)
            out_logits = self.forward(src, tgt)   #(N,T,output_size)
            # print("out_logits",out_logits.shape,out_logits[:,:t+1,0])
            # save the last token step to output
            outputs[:,t,:] = out_logits[:,t,:]
            # print(outputs.shape)

            # Next token using greedy pick on last step of the output
            best_token = out_logits[:,t,:].argmax(dim=1)   # shape (N,)
            #update tgt
            if t<(T-1):
                tgt[:,t+1]  = best_token

        return outputs



    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


### Ed Discussion and https://static.us.edusercontent.com/files/SFpcXmcBkJZWjbOgM2HJUXpZ
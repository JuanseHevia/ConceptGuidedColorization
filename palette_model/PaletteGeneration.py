# Code for U Net Encoder-Decoder architecture to encode and decode
# textual input into paltte representation

import torch.nn as nn
import torch.functional as F
import open_clip
import torch

class EncoderBlock(nn.Module):
    """
    Basic building block for the encoder.
    Takes in CLIP embeddings and reduces the dimensionality.
    """

    def __init__(self, in_channels, out_channels, num_layers=3,
                 activation='relu', BASE_DIM=None):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList()
        _in_channels = BASE_DIM if BASE_DIM is not None else in_channels
        
        if BASE_DIM:
            self.layers.append(nn.Linear(in_channels, BASE_DIM))

        for _l in range(num_layers):
            self.layers.append(nn.Linear(_in_channels, _in_channels))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(_in_channels, out_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        

class TextCLIPEncoder(nn.Module):
    """
    Takes in textual input and encodes it into a latent representation.
    We use CLIP embeddings to convert text into a latent representation.
    """

    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', 
                 n_blocks=4, in_channels=512, reduce_factor=2,):
        super(TextCLIPEncoder, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # turn into evaluation mode, freeze weights
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.layers = nn.ModuleList()
        _out_internal_channels = in_channels // reduce_factor
        _in_dim = in_channels

        for _b in range(n_blocks):
            self.layers.append(EncoderBlock(_in_dim, _out_internal_channels))
            _in_dim = _out_internal_channels
            _out_internal_channels //= reduce_factor
                  

    def get_embedding(self, text):
        _text = self.tokenizer(text)
        with torch.no_grad():
            text_features = self.model.encode_text(_text)
        return text_features

    def forward(self, x):
        """
        Compute forward pass.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder to convert the latent representation back to palette representation.
    """

    def __init__(self, in_channels, out_channels, num_layers=3, activation='relu'):
        super(DecoderBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _l in range(num_layers):
            self.layers.append(nn.Linear(in_channels, in_channels))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class PaletteDecoder(nn.Module):
    """
    Takes in latent representation and decodes it into palette representation.
    """

    def __init__(self, n_blocks=4, in_channels=512, num_colors=5, reduce_factor=2):
        super(PaletteDecoder, self).__init__()
        self.layers = nn.ModuleList()
        _in_dim = in_channels

        for _b in range(n_blocks):
            _out_internal_channels = _in_dim // reduce_factor
            self.layers.append(DecoderBlock(_in_dim, _out_internal_channels))
            _in_dim = _out_internal_channels

        out_channels = num_colors * 3 # num_colors * RGB channels

        self.final_layer = nn.Linear(_in_dim, out_channels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x).view(-1, 5, 3)
    
    def get_arch(self):
        """
        Print the architecture of the model.
        Focus on layer values, input and output dimensions.
        """
        print(self)
        for layer in self.layers:
            print(layer)
        print(self.final_layer)
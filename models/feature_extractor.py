import torch
from torch import nn
import numpy as np
import math

class feature_extractor_spp(nn.Module):
    def __init__(self, in_features=1, encoder_h=32, enc_width=(3, 3, 3, 3, 3, 3),
             dropout=(0., 0., 0.5, 0., 0., 0.5), enc_downsample=(1, 1, 2, 1, 1, 2), embedding_dim=100):
        super(feature_extractor_spp, self).__init__()
        self.output_num = [5, 3, 2]
        self.encoder = nn.Sequential()
        for i, (width, downsample, drop) in enumerate(zip(enc_width, enc_downsample, dropout)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv2d(in_features, encoder_h, (1,width), stride=(1,downsample), padding=width // 2),
                nn.Dropout2d(drop),
                nn.GroupNorm(encoder_h, encoder_h),   
                nn.ReLU(),
            ))
            in_features = encoder_h
        self.linear = nn.Linear(sum(encoder_h * [i*i for i in self.output_num]), embedding_dim)

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int(math.ceil((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2))
            w_pad = int(math.ceil((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2))
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
            else:
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp
    
    def forward(self, x):
        bs = x.shape[0]
        out = self.encoder(x)
        out = self.spatial_pyramid_pool(out,bs,[int(out.size(2)), int(out.size(3))],self.output_num)
        out = self.linear(out)
        return out
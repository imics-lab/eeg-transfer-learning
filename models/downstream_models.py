import torch
from torch import nn
from .feature_extractor import feature_extractor_spp

class Supervised_EEGBCI(nn.Module):
    def __init__(self, in_features=1, encoder_h=32, num_classes = 1, enc_width=(3, 3, 3, 3, 3, 3),
                 dropout=(0., 0., 0.5, 0., 0., 0.5), enc_downsample=(1, 1, 2, 1, 1, 2), embedding_dim=100):
        super().__init__()
        self.encoder = nn.Sequential()
        for i, (width, downsample, drop) in enumerate(zip(enc_width, enc_downsample, dropout)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv2d(in_features, encoder_h, (1,width), stride=(1,downsample), padding=(1, width // 2)),
                nn.Dropout2d(drop),
                nn.GroupNorm(encoder_h, encoder_h),   
                nn.ReLU(),
            ))
            in_features = encoder_h
            
        self.maxpool = nn.MaxPool2d(kernel_size=(3,10)) #output (N, 32, 10, 20)
        self.flatten = nn.Flatten() # (N, 6400)
        self.linear1 = nn.Linear(6400, 1000)
        self.linear2 = nn.Linear(1000, 100)
        self.output = nn.Linear(100, num_classes)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.output(out)
        out = torch.sigmoid(out)
        
        return out


class Supervised_TUH(nn.Module):
    def __init__(self, in_features=1, encoder_h=32, num_classes = 1, enc_width=(3, 3, 3, 3, 3, 3),
                 dropout=(0., 0., 0.5, 0., 0., 0.5), enc_downsample=(1, 1, 2, 1, 1, 2), embedding_dim=100):
        super().__init__()
        self.feature_extractor = feature_extractor_spp(in_features, encoder_h, enc_width, dropout, enc_downsample, embedding_dim).cuda()
        self.linear = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.linear(features)
        out = torch.sigmoid(out)
        
        return out
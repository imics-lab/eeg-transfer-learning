import torch
from torch import nn
from .feature_extractor import feature_extractor_spp

class SSL_model(nn.Module):
    def __init__(self, in_features=1, encoder_h=32, enc_width=(3, 3, 3, 3, 3, 3),
                 dropout=(0., 0., 0.5, 0., 0., 0.5), enc_downsample=(1, 1, 2, 1, 1, 2), embedding_dim=100):
        super().__init__()
        self.feature_extractor = feature_extractor_spp(in_features, encoder_h, enc_width, dropout, enc_downsample, embedding_dim).cuda()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x1, x2):
        
        h_first = self.feature_extractor(x1)
        h_second = self.feature_extractor(x2)

        h_combined = torch.abs(h_first - h_second)

        out = self.linear(h_combined)
        return out
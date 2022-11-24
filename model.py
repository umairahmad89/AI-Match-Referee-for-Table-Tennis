import torch.nn as nn
import torch.autograd

import numpy as np


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
#         print("base mein")
        self.L1 = nn.Sequential(nn.Conv2d(3, 32, 5), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))  # 32 * 98 * 98
        self.L2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))  # 64 * 48 * 48
        self.L3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 * 23 * 23
        self.L4 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU())  # 128 * 21 * 21
        self.L5 = nn.Sequential(nn.Conv2d(128, 32, 3), nn.BatchNorm2d(32), nn.ReLU())  # 4 * 21 * 21
        

    def forward(self, x):
        x = self.L5(self.L4(self.L3(self.L2(self.L1(x)))))
        b,c,h,w = x.shape
        return x

    
class BallDetectionGS(nn.Module):
    def __init__(self):
        super(BallDetectionGS, self).__init__()
        self.base_block_gs = BaseNet()
        self.fc2_bd_gs = nn.Linear(16, 2)
        _x = self._get_conv_output((3,320,128))
        self.fc1_gs = nn.Sequential(
            nn.Linear(_x, 256),
            nn.ReLU(), 
            nn.Linear(256, 16), 
            nn.ReLU()
            )

    def forward(self, x):
        features = self.base_block_gs(x)
        _, c, w, h = features.shape
        x = features.view(-1, c*h*w)
        x = self.fc1_gs(x)
        return self.fc2_bd_gs(x), features

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.base_block_gs(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
class BallDetectionLS(nn.Module):

    def __init__(self) -> None:
        super(BallDetectionLS, self).__init__()
        self.base_block_ls = BaseNet()
        self.fc2_bs_ls = nn.Linear(16, 2)
        _x = self._get_conv_output((3,320,128))
#         print(_x)
        self.fc1_ls = nn.Sequential(
            nn.Linear(_x, 256),
            nn.ReLU(), 
            nn.Linear(256, 16), 
            nn.ReLU()
            )
        

    def forward(self, x):
        features = self.base_block_ls(x)
        _,c,h,w = features.shape
        x = features.view(-1, c*h*w)
        x = self.fc1_ls(x)
        return self.fc2_bs_ls(x), features

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.base_block_ls(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

class EventSpotting(nn.Module):
    def __init__(self):
        super(EventSpotting, self).__init__()
        self.ball_gs = BallDetectionGS()
        self.ball_ls = BallDetectionLS()
        self.L6 = nn.Sequential(nn.Conv2d(32, 16, 3), nn.BatchNorm2d(16), nn.ReLU())
        self.L7 = nn.Sequential(nn.Conv2d(16, 8, 3), nn.BatchNorm2d(8), nn.ReLU())
        _x = self._get_conv_output((3,320,128))
        self.fc_es = nn.Sequential(nn.Linear(_x, 16))
        self.Last_es = nn.Linear(16, 4)

    def forward(self, local_features, global_features):
        x = local_features+global_features
        x = self.L7(self.L6(x))
        b,c,h,w = x.shape
        return self.Last_es(self.fc_es(x.view(-1,c*h*w)))


    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        _, output_feat = self.ball_gs(input)
        output_feat = self.L7(self.L6(output_feat))
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size


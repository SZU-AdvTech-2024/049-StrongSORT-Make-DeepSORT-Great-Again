import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from functools import partial

import sys
sys.path.append('dinov2')

from dinov2.eval.linear import create_linear_input
from dinov2.eval.linear import LinearClassifier
from dinov2.eval.utils import ModelWithIntermediateLayers


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # get feature model
        # model = torch.hub.load(
        #     "facebookresearch/dinov2", 'dinov2_vitg14', pretrained=True
        # ).to(device)
        model = torch.hub.load(r'/home/kaiwen/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', source='local').cuda()
        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float16
        )
        self.backbone = ModelWithIntermediateLayers(
            model, n_last_blocks=6, autocast_ctx=autocast_ctx
        ).to(device)

        # with torch.no_grad():
        #     sample_input = torch.randn(1, 3, 224, 224).to(device)
        #     sample_output = self.feature_model(sample_input)

        # get linear readout
        # out_dim = create_linear_input(
        #     sample_output, use_n_blocks=1, use_avgpool=True
        # ).shape[1]
        # self.classifier = LinearClassifier(
        #     out_dim, use_n_blocks=1, use_avgpool=True, num_classes=751
        # ).to(device)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.classifier(x)
        return x
    
    def load_param(self, device='cuda:0'):
        model_weights = '/mnt/data/kaiwen/code/deepsort/StrongSORT/others/ckpt_g.t7'
        state_dict = torch.load(model_weights, map_location=lambda storage, loc: storage)['net_dict']
        self.backbone.load_state_dict(state_dict,False)
        
        




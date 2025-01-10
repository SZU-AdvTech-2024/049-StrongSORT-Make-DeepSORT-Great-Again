import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load
import torchvision.models as models
from functools import partial

dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}

class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=True, backbone = 'dinov2_g', head = 'linear', backbones = dino_backbones):
        super(Net, self).__init__()
        self.heads = {
            'linear':linear_head,
        }
        self.backbones = dino_backbones
        self.backbone = torch.hub.load(r'/home/kaiwen/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitg14', source='local').cuda()
        # self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').cuda()
        self.backbone.eval()
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)
        self.reid = reid

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            if self.reid:
                return x
        x = self.head(x)
        return x




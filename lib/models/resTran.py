import pdb

import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.nn as nn
import torch.nn.functional as F
from lib.models.attention import *
#from lib.models.backbone import ResTranformer
from lib.models.model import Model
import lib.models.resnet as resnet
import os

class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = config.model_vision_loss_weight  # ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = 512 # ifnone(config.model_vision_d_model, 512)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else:
            self.backbone = resnet.resnet34()

        if config.model_vision_attention == 'position':
            mode = 'nearest'  # ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8 * 32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, config.NUM_CLASSES+1)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)
        # print(features.size())
        # pdb.set_trace()
        features=features.transpose(0,1)
        attn_vecs = self.attention(features)  # (N, T, E), (N, T, H, W)
        #attn_vecs, _ = self.attention(features)  # (N, T, E), (N, T, H, W)
        #logits=attn_vecs
        logits = self.cls(attn_vecs)  # (N, T, C)
        a,b,c=logits.size()
        #logits = logits.transpose(0, 1)
        logits=F.log_softmax(logits, dim=2)

        pt_lengths = self._get_length(logits)
        #pdb.set_trace()
        return logits
        # return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
        #'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision'}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

from collections import OrderedDict
def load(model, file, device=None):
    if device is None:
        device = 'cpu'
    elif isinstance(device, int):
        device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)['state_dict']
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    newstate=OrderedDict()
    for k,v in state.items():
        #print(k)
        # if 'transformer' in k:
        #      continue
        #newstate[k.replace('resnet.','')]=v
        newstate[k] = v
    # #print(state.keys())
    model.load_state_dict(newstate,strict=True)
    return model

def get_resTran(config):
    model = BaseVision(config)
    model = load(model,'/mnt/model_checkpoint/synthtext_checkpoint_5_acc_18.pth')
    #model = load(model,'/root/zm/attention_CRNN_pretrain/licenseoutput/OWN/crnn/2022-06-17-10-43/checkpoints/checkpoint_100_acc_0.0602.pth')
    #model = CRNN(64, 3, 85, 512)
    print("config.NUM_CLASSES:",config.NUM_CLASSES)
    #model.apply(weights_init)

    return model
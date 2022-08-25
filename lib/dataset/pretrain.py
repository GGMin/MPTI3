from __future__ import print_function, absolute_import

import pdb
import random
import torch.utils.data as data
import os
import numpy as np
import cv2
import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image, ImageFont, ImageDraw

np.set_printoptions(threshold=np.inf)

class pretrain(VisionDataset):
    def __init__(self, config, transform=None, target_transform=None, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.dataset_name = config.DATASET.DATASET
        self.config=config
        txt_file = config.pretrainfilename
        self.labels=[]
        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8',errors='ignore') as file:
            for c in file.readlines():
                # print(c)
                # pdb.set_trace()
                self.labels.append(c.replace('\n',' ').strip())
        print(self.labels)
        print("load {} images!".format(self.__len__()))

        has_separate_transform = transform is not None or target_transform is not None
        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms=transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # img_nameo = self.labels[idx].keys()
        # for item in img_nameo:
        #     img_name=item
        #
        # img = Image.open(img_name)
        #print(self.labels[idx])
        img= CreateImg(self.labels[idx])
        img= self.transforms(img)
        #img=torch.permute(img, (2,0,1))
        #pdb.set_trace()

        #pdb.set_trace()
        return img, idx


class StandardTransform(object):
    def __init__(self, transform= None, target_transform= None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        return input

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


def CreateImg(text):
    imageH=random.randrange(32,128,4)
    imageW=random.randrange(64,256,4)
    fontSize = random.randrange(imageH/2,imageH-4)
    R=random.randrange(1,254)
    G = random.randrange(1, 254)
    B = random.randrange(1, 254)
    textpointH=random.randrange(0,int(imageH/2))
    textpointW = random.randrange(0, int(imageW/2))
    # »­²¼ÑÕÉ«
    im = Image.new("RGB", (imageW, imageH), (R, G, B))
    dr = ImageDraw.Draw(im)
    #print(text)
    Font=ImageFont.truetype("/mnt/arial.ttf", fontSize)
    # ÎÄ×ÖÑÕÉ«
    dr.text((textpointW, textpointH), text, font=Font, fill=(255-R,255-G,255-B))
    return im



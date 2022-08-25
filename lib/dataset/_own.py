from __future__ import print_function, absolute_import

import pdb

import torch.utils.data as data
import os
import numpy as np
import cv2
import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple

np.set_printoptions(threshold=np.inf)

class _OWN(VisionDataset):
    def __init__(self, config, transform=None, target_transform=None, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)
        self.config=config
        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']
        self.labels=[]
        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8-sig',errors='ignore') as file:
            # for c in file.readlines():
            #     temp=(c.strip().split(' ')[1:])[0]
            #     list=''
            #     for tem in temp:
            #         list=list+tem+'>'
            #     self.labels.append({c.split(' ')[0]: ['>'+list]})
            self.labels = [{c.split(' ')[0]: c.strip().split(' ')[1:]} for c in file.readlines()]
        #print(self.labels)
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

        img_nameo = self.labels[idx].keys()
        for item in img_nameo:
            img_name=item

        img = Image.open(img_name)
        # #print(os.path.join(self.root, img_name))
        # img = cv2.imread(img_name)
        # #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_h, img_w,_ = img.shape
        # img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        # img = np.reshape(img, (self.inp_h, self.inp_w, 3))
        #
        # img = img.astype(np.float32)
        # img = (img/255. - self.mean) / self.std
        # img = img.transpose([2, 0, 1])
        # img = torch.from_numpy(img)
        # print(img.size())
        img= self.transforms(img)
        #img=torch.permute(img, (2,0,1))
        #print(img.size())

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






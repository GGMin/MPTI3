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
from torchvision import datasets
import re
from torchvision.transforms import functional as F

np.set_printoptions(threshold=np.inf)


class mysynthtext(datasets.ImageFolder):
    def __init__(self, config, transform, target_transform=None, is_train=True):
        #super(datasets.ImageFolder,self).__init__(config.mysynthtext.root,Image.open)
        self.root=config.mysynthtext.root
        self.samples=[]
        self.transform = transform
        for root, dirs, files in os.walk(self.root):
            #print(root,files)
            for name in files:
                self.samples.append(os.path.join(root,name))
        self.img=Image.open('/mnt/mjsynthtext/ramdisk/max/90kDICT32px/1/6/471_HALON_34663.jpg')
        self.idx='HALON'
        self.img=self.transform(self.img)
        #print(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path= self.samples[index]
        pattern = r'[_|.]'
        target = re.split(pattern, path)[1]
        #print(path)
        #pdb.set_trace()
        try:
            img = Image.open(path)
            # pdb.set_trace()
            img = self.transform(img)
        except:
            print(path)
            target=self.idx
            img=self.img
        return img, target










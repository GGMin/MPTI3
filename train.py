import argparse
import pdb

from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.resTran as resTran
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info
import torch.nn as nn
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int

class Mae(torch.nn.Module):
    def __init__(self, maskratio,patch_h,patch_w):
        super().__init__()
        self.maskratio=maskratio
        self.patch_h=patch_h
        self.patch_w=patch_w

    def forward(self, img):
        c,h,w= img.size()
        #print(img)
        num_patches=(h // self.patch_h) * (w // self.patch_w)
        x= img.view(
            c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(1, 3, 2, 4, 0).reshape(num_patches,-1)
        _, length=x.size()
        for i in range(num_patches):
            if random.randint(1,10) < 10*self.maskratio:
                x[i,:]=torch.randn((1,length))
        x=x.view(h // self.patch_h, w // self.patch_w, self.patch_h, self.patch_w, c).permute(4,0,2,1,3).reshape(c,h,w)
        return x


class randomBatchResize(torch.nn.Module):
    def __init__(self, minsize, maxsize, batch, interpolation=InterpolationMode.BILINEAR, antialias=None):
        super().__init__()

        self.min_size = minsize
        self.max_size = maxsize
        self.count=0
        self.batch=batch
        self.size=minsize

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if self.count != self.batch:
            self.count=self.count+1
            print(self.count)
        else:
            self.count=0

            self.size=(random.randrange(self.min_size[0],self.max_size[0],4), random.randrange(self.min_size[1],self.max_size[1],4))

        return F.resize(img, self.size, self.interpolation, None, self.antialias)



def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    print(img.shape)
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape

    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    print(img.shape)
    np.set_printoptions(threshold=np.inf)
    print(img)
    img = torch.from_numpy(img)


    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    print(preds.data)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()
    print(args.cfg)
    with open(args.cfg) as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():

    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = resTran.get_resTran(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
        print("cuda!!!!!!!!!")
    else:
        device = torch.device("cpu:0")

    #device = torch.device('cpu')
    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    #model_info(model)

    train_transform = transforms.Compose([
        #randomBatchResize((104,108),(104,108),config.TRAIN.BATCH_SIZE_PER_GPU),
        #randomBatchResize((32, 128), (160, 160), config.TRAIN.BATCH_SIZE_PER_GPU),
        transforms.Resize((config.imagesize.w, config.imagesize.h)),
        transforms.RandomRotation((-15, 15)),
        #transforms.GaussianBlur((3, 5)),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        Mae(0.7,16,16),
        transforms.Normalize(config.DATASET.MEAN, config.DATASET.STD),
    ])
    val_transform= transforms.Compose([
        transforms.Resize((config.imagesize.w,config.imagesize.h)),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        #Mae(0.3),
        transforms.Normalize(config.DATASET.MEAN, config.DATASET.STD),
    ])

    train_dataset = get_dataset(config)(config, train_transform, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, val_transform, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    softmax=nn.Softmax(dim=0)

    count=0
    best_acc = 0
    print(config.DATASET.ALPHABETS)
    print("blank= ",config.DATASET.ALPHABETS[84])
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        #pdb.set_trace()
        lr_scheduler.step()
        if count==5:
            print("##every 5 epoch one validation##")
            print("################################")
            acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict,softmax)

            is_best = acc > best_acc
            best_acc = max(acc, best_acc)

            print("is best:", is_best)
            print("best acc is:", best_acc)
            # save checkpoint
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "best_acc": best_acc,
                },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            )
            count=0
        count=count+1

    writer_dict['writer'].close()

if __name__ == '__main__':

    main()
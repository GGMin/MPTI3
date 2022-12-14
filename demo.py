import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import torch.nn as nn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
from lib.utils.utils import model_info

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib\config\OWN_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/Cars27.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default=r'./licenseoutput/checkpoint_80_acc_0.0000.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        #print(f)
        config = yaml.load(f,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    #h, w = img.shape
    print(img.shape)
    # fisrt step: resize the height and width of image to (32, x)
    #img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape


    # w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    # img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))
    img_h, img_w = img.shape
    inp_h = config.MODEL.IMAGE_SIZE.H
    inp_w = config.MODEL.IMAGE_SIZE.W
    img = cv2.resize(img, (0, 0), fx=inp_w / img_w, fy=inp_h / img_h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (inp_h, inp_w, 1))
    print(img.shape)
    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    #print(img.shape)
    np.set_printoptions(threshold=np.inf)
    #print(img)
    img = torch.from_numpy(img)


    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    m = nn.Softmax(dim=0)
    #preds=m(preds)
    print(preds.shape)
    print(preds[0])
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    print(preds.data)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda')
    #print(config)
    model = crnn.get_crnn(config).to(device)
    model_info(model)
    #model.load_state_dict(torch.load(args.checkpoint))
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)


    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    started = time.time()
    recognition(config, img, model, converter, device)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))


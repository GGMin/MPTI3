from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch
import pdb
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def minDistance(word1: str, word2: str) -> int:
    n = len(word1)
    m = len(word2)

    # ÓÐÒ»¸ö×Ö·û´®Îª¿Õ´®
    if n * m == 0:
        return n + m

    # DP Êý×é
    D = [[0] * (m + 1) for _ in range(n + 1)]

    # ±ß½ç×´Ì¬³õÊ¼»¯
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    # ¼ÆËãËùÓÐ DP Öµ
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)

    return D[n][m]


def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time

        data_time.update(time.time() - end)
        #print(idx)
        labels = utils.get_batch_label(dataset, idx,config.DATASET.DATASET)
        inp = inp.to(device)
        #print(inp.size())
        #print(torch.isnan(inp))
        # inference
        preds = model(inp)
        preds = preds.to(torch.float64)

        # compute loss
        batch_size = inp.size(0)
        #print(labels)
        text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        #print(preds)
        #pdb.set_trace()
        #print(preds_size)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        #print(preds_size)
        loss = criterion(preds, text, preds_size, length)

        # if torch.isnan(inp).any() or torch.isinf(inp).any():
        #     print("AAAAAAA!",idx,inp)
        # if torch.isnan(preds).any() or torch.isinf(preds).any():
        #     print("BBBBBBBB!",idx,preds)
        #     raise(idx)
        # if torch.isnan(text).any() or torch.isinf(text).any():
        #     print("CCCCCCC!",idx,preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict,softmax):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    character_correct=0
    character_sum=0

    count=0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):
            count=count+1
            labels = utils.get_batch_label(dataset, idx, config.DATASET.DATASET)
            inp = inp.to(device)
            if config.DATASET.DATASET == 'pretrain':
                if count>500:
                    break
            # model.eval()
            # preds = model(inp)
            # #print(preds.shape)
            # _, preds = preds.max(2)
            # preds = preds.transpose(1, 0).contiguous().view(-1)
            # print(preds.data)
            # pdb.set_trace()

            # inference
            #print(inp.shape)
            preds = model(inp)
            #print(preds.shape)
            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            # #print(preds)
            # print(preds.shape)
            # print(preds[0][0])
            # preds, _= preds.max(2)
            # print(preds)
            #
            # pdb.set_trace()
            # preds = preds.transpose(1, 0).contiguous().view(-1)
            # print(preds.data)

            #preds=softmax(preds)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)


            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                # print('pred:', sim_preds)
                # print('target:', labels)
                targ=''
                for i in target:
                    targ=targ+i
                #print(targ)
                sameC_distance=minDistance(pred, targ)
                character_correct=character_correct+len(target)-sameC_distance
                character_sum=character_sum+len(target)
                if pred == targ:
                    n_correct += 1

            # if (i + 1) % config.PRINT_FREQ == 0:
            #     print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            # if i == config.TEST.NUM_TEST_BATCH:
            #     break
    # print(preds.data)
    # pdb.set_trace()
    raw_preds = converter.decode(preds.data, preds_size.data, raw=False)[:config.TEST.NUM_TEST_DISP]
    #print(raw_preds)
    print('results: {0}'.format(raw_preds))
    #pdb.set_trace()
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    #pdb.set_trace()
    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("WAR  [#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    print("CAR  [#correct:{} / #total:{}]".format(character_correct, character_sum))
    accuracy = n_correct / float(num_test_sample)
    CAR_accuracy = character_correct / character_sum
    print('Test loss: {:.4f}, WAR_accuray: {:.4f}, CAR_accuracy: {:.4f}'.format(losses.avg, accuracy, CAR_accuracy))


    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy
#!/usr/bin/python3 #coding=utf-8

import os
import sys
import datetime
import numpy as np

import torch
import time
import torch.nn as nn
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from tensorboardX import SummaryWriter
    BOARD_FLAG = True
except:
    BOARD_FLAG = False

from lib import dataset
from lib.dataset import train_collate_fn
from network  import Segment
import logging as logger
from lib.data_prefetcher import DataPrefetcher
import argparse
import pytorch_iou
from lib.utils import load_model

DATA_PATH = "./data/RGBD_sal/train"
TEST_PATH = [
           './data/RGBD_sal/test_in_train/test_in_train']


writer = SummaryWriter('./write'+'summary')
best_mae = 1
best_fscore = 0.1
best_epoch = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)  # 修改
    torch.backends.cudnn.deterministic = True

conf = argparse.ArgumentParser(description="train model")
conf.add_argument("--tag", default='res50', type=str)
conf.add_argument("--savepath", type=str, default='./best', help="where to save models?")
conf.add_argument("--lr", type=float, default=0.05)
conf.add_argument("--bz_size", type=int, default=20)  # 修改部分 bz_size gpu
conf.add_argument("--epochs", type=int, default=150)
conf.add_argument("--decay", type=float, default=1e-4)
conf.add_argument("--seed", type=int, default=1997)
conf.add_argument("--gpu", type=str, default="0")
conf.add_argument("--momen", type=str, default="0.9")
conf.add_argument("--model_pt", type=str, default="None")
args = conf.parse_args()
setup_seed(args.seed)
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

cfg = dataset.Config(datapath=DATA_PATH, savepath=args.savepath, mode='train', batch=args.bz_size, lr=args.lr,
                     momen=0.9, decay=args.decay, epoch=args.epochs, train_scales=[224, 256, 320])
data = dataset.RGBDData(cfg)
train_loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=0, drop_last=True,
                    collate_fn=train_collate_fn)

# network
net = Segment(backbone='resnet50')
IOU = pytorch_iou.IOU(size_average=True)

if os.path.exists(args.model_pt):
    msg = "Loading pretrained model_pt:%s" % args.model_pt
    print(msg)
    logger.info(msg)
    net.load_state_dict(torch.load(args.model_pt), strict=False)



## parameter


for e in TEST_PATH:
    cfg = dataset.Config(datapath=e, epoch=args.epochs, savepath=args.savepath, mode='test')
    # tag = conf.tag
    data = dataset.RGBDData(cfg)
    test_loader = DataLoader(data, batch_size=2, shuffle=True, num_workers=0)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="train_%s.log" % (args.tag), filemode="w")
logger.info("Configuration:{}".format(args))
logger.info("SEED:%s, gpu:%s" % (args.seed, args.gpu))


def train(train_loader,net,optimizer,epoch):
    # dataset
    net.train()
    net = nn.DataParallel(net)
    net.cuda()

    if BOARD_FLAG:
        sw = SummaryWriter(cfg.savepath)
    global_step = 0
    epochloss = 0

    step = -1
    prefetcher = DataPrefetcher(train_loader, cnt=3)
    image, depth,  mask, gate_gt = prefetcher.next()
    while image is not None:
        out, side_fusion3,side_fusion4,y1,y2,y3,gate = net.forward(image, depth)
        # dominant loss
        dom_loss = F.binary_cross_entropy_with_logits(out, mask) + IOU(y1, mask)
        # aux. loss
        # loss2_1 = F.binary_cross_entropy_with_logits(out2_1, mask)
        # loss3_1 = F.binary_cross_entropy_with_logits(out3_1, mask)
        # loss4_1 = F.binary_cross_entropy_with_logits(out4_1, mask)
        # loss5_1 = F.binary_cross_entropy_with_logits(out5_1, mask)
        # loss2_2 = F.binary_cross_entropy_with_logits(out2_2, mask)
        # loss3_2 = F.binary_cross_entropy_with_logits(out3_2, mask)
        # loss4_2 = F.binary_cross_entropy_with_logits(out4_2, mask)
        # loss5_2 = F.binary_cross_entropy_with_logits(out5_2, mask)
        lossside_output3 = F.binary_cross_entropy_with_logits(side_fusion3,mask) + IOU(y2,mask)
        lossside_output4 = F.binary_cross_entropy_with_logits(side_fusion4,mask) + IOU(y3,mask)
        # regression
        reg_loss = F.smooth_l1_loss(gate, gate_gt) * 2
        loss = dom_loss + reg_loss + lossside_output3* 0.5 +lossside_output4 * 0.4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        global_step += 1
        step += 1
        if BOARD_FLAG:
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'dom_loss':dom_loss.item(), 'reg_loss':reg_loss.item(), 'loss':loss.item()}, global_step=global_step)
        if global_step % 1 == 0:
            msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | dom_loss=%.6f  reg_loss=%.6f'%(datetime.datetime.now(),  global_step, epoch, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(),  dom_loss.item(), reg_loss.item())
            print(msg)
            logger.info(msg)
            epochloss = epochloss+loss.item()
            print(epochloss)
            print(global_step)
            if global_step % 328 == 0:
                print(global_step)
                MAEloss = epochloss/ 328
                print(MAEloss)
                msg  = 'step:%d/%d/%d | lr=%.6f | epochloss=%.6f | MAEloss=%.6f'%(global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], epochloss , MAEloss)
                print(msg)
                logger.info(msg)
                epochloss = 0
        image, depth, mask, gate_gt = prefetcher.next()

    if (epoch+1) in range(145,151): # or dom_loss.item() <= 0.025:
        #logger.info("saving model-%s ..., loss:%s"%(epoch+1, dom_loss.item()))
        torch.save(net.module.state_dict(), cfg.savepath+'/modal'+'/model-'+str(epoch+1))



# class Test(object):
#     def __init__(self, test_loader, net, epoch):
#         ## dataset
#         #self.cfg    = Dataset.Config(datapath='../data/SOD', snapshot='./out/model-30', mode='test')
#
#
#         self.datapath = datapath.split("/")[-1]
#         print("Testing on %s"%self.datapath)
#         self.epoch = epoch
#         self.cfg = Dataset.Config(datapath = datapath, epoch=self.epoch,savepath=conf.savepath, mode='test')
#         self.tag = conf.tag
#         self.data   = Dataset.RGBDData(self.cfg)
#         self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=0)
#         ## network
#         self.net    =  net(backbone='resnet50', cfg=self.cfg)
#         self.net.train(False)
#         self.net.cuda()
#         self.net.eval()


def Test (test_loader, net, epoch):
    net.eval()
    global best_mae, best_epoch
    with torch.no_grad():
        mae, fscore, cnt, number   = 0, 0, 0, 256
        mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
        cost_time = 0
        for image, d, mask, (H,W), name in test_loader:
            image, d, mask  = image.cuda().float(), d.cuda().float(), mask.cuda().float()
            start_time = time.time()
            out, gate = net(image, d)
            pred  = torch.sigmoid(out)
            torch.cuda.synchronize()
            end_time = time.time()
            cost_time += end_time - start_time

            ## MAE
            #pred     = F.interpolate(pred, size=(H,W), mode='bilinear')
            #mask     = F.interpolate(mask, size=(H,W), mode='bilinear')

            cnt += 1
            mae += (pred-mask).abs().mean()
            ## F-Score
            precision = torch.zeros(number)
            recall    = torch.zeros(number)
            for i in range(number):
                temp         = (pred >= threshod[i]).float()
                precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)

            mean_pr += precision
            mean_re += recall
            best_MAE = mae/cnt
        # fps = len(self.loader.dataset) / cost_time
        # msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)

        if epoch == 1:
            best_mae = best_MAE
        else:
            if best_MAE < best_mae:
                best_mae = best_MAE
                best_epoch = epoch
                torch.save(net.state_dict(), cfg.savepath + '/modal/'+'EGANet_epoch_best.pth')
                print('best epoch:{}'.format(best_epoch))
                msg = 'MAE=%.6f' % (best_mae)
                print(msg)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, best_MAE, best_mae, best_epoch))


            # logger.info(msg)


if __name__=='__main__':
    for epoch in range(1,151):
        rgb_base, rgbd_base, d_base, head = [], [], [], []
        fc_params = []
        for name, param in net.named_parameters():
            if 'bkbone_rgbd' in name:
                rgbd_base.append(param)
            elif 'bkbone_rgb' in name:
                rgb_base.append(param)
            elif 'bkbone_d' in name:
                d_base.append(param)
            elif 'fc' in name:
                fc_params.append(param)
            else:
                head.append(param)
        assert len(rgbd_base) == 0
        optimizer = torch.optim.SGD([{'params': rgb_base}, {'params': d_base}, {'params': fc_params}, {'params': head}],
                                    lr=0.05, momentum=0.9, weight_decay=args.decay)  # nesterov=True)
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (150 + 1) * 2 - 1)) * 0.05* 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (150 + 1) * 2 - 1)) * 0.05* 0.1
        optimizer.param_groups[2]['lr'] = (1 - abs((epoch + 1) / (150 + 1) * 2 - 1)) * 0.05
        optimizer.param_groups[3]['lr'] = (1 - abs((epoch + 1) / (150 + 1) * 2 - 1)) * 0.05

        train(train_loader,net,optimizer, epoch)
        Test(test_loader, net, epoch)

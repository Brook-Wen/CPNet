import os
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
import scipy.misc as sm
import torchvision.utils as vutils
from torch.backends import cudnn
from torch.autograd import Variable
from torch.optim import Adam, SGD, lr_scheduler
from torch.nn import utils, functional as F
from collections import OrderedDict
from model.model import build_model, weights_init
from model.loss_functions import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import visdom
vis = visdom.Visdom(env='CPN')


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        # self.config.cuda = False # only test
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [60,]
        self.pretrained = True
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model), strict=False)
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'), strict=False)
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(base_model_cfg=self.config.arch)
        self.criterion = BCELoss(reduction='mean')
        self.criterion1 = DiceLoss(reduction='mean')

        if self.config.cuda:
            self.net = self.net.cuda()
            self.criterion = self.criterion.cuda()
            self.criterion1 = self.criterion1.cuda()

        if self.config.mode == 'train':
            self.net.train()
            # self.net.eval()  # use_global_stats = True
        elif self.config.mode == 'test':
            self.net.eval()
        
        self.net.apply(weights_init)

        self.lr = self.config.lr
        self.betas = self.config.betas
        self.eps = self.config.eps
        self.wd = self.config.wd
        if self.pretrained:
            if self.config.arch == 'resnet' or self.config.arch == 'vgg':
                if self.config.load == '':
                    pretrained_dict = torch.load(self.config.pretrained_model)
                    model_dict = self.net.base.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.net.base.load_state_dict(model_dict, strict=False)
                else:
                    self.net.load_state_dict(torch.load(self.config.load), strict=False)
                
                base_params = list(map(id, self.net.base.parameters()))
                sub_params = filter(lambda p: id(p) not in base_params, self.net.parameters())
                sub_params = filter(lambda p: p.requires_grad, sub_params)
                self.optimizer = Adam([{'params': self.net.base.parameters()},
                                    {'params': sub_params, 'lr': self.lr*10}],
                                    lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wd)
            else:
                raise AssertionError
        else:
            self.optimizer = Adam(self.net.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wd)

        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O1")
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_epoch, gamma=0.1)
        self.print_network(self.net, 'Net Structure')

    def test(self):
        for i, data_batch in enumerate(self.test_loader):
            images, name = data_batch['image'], data_batch['name'][0]
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()

                sal_pred, _, _ = self.net(images)
                pred = np.squeeze(torch.sigmoid(sal_pred).cpu().data.numpy())
                pred = 255 * pred
                if not os.path.exists(self.config.test_fold): os.mkdir(self.config.test_fold)
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '.png'), pred)
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        for epoch in range(self.config.epoch):
            loss_f, loss_s, loss_e, loss_t = 0, 0, 0, 0
            self.optimizer.zero_grad()
            self.net.zero_grad()

            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge']

                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label = Variable(sal_image), Variable(sal_label)
                sal_edge = Variable(sal_edge)
                if self.config.cuda:
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()
                    sal_edge = sal_edge.cuda()

                sal_pred, s_pred, e_pred = self.net(sal_image)

                f_loss = self.criterion(sal_pred, sal_label) + self.criterion1(sal_pred, sal_label)
                loss1 = f_loss / (self.iter_size * self.config.batch_size)
                loss_f += loss1.item()

                for l in range(len(s_pred)):
                    if l==0:
                        s_loss = self.criterion(s_pred[l], sal_label) + self.criterion1(sal_pred, sal_label)
                    else:
                        s_loss += self.criterion(s_pred[l], sal_label) + self.criterion1(sal_pred, sal_label)
                loss2 = s_loss / (self.iter_size * self.config.batch_size)
                loss_s += loss2.item()

                for l in range(len(e_pred)):
                    if l==0:
                        e_loss = self.criterion1(e_pred[l], sal_edge)
                    else:
                        e_loss += self.criterion1(e_pred[l], sal_edge)
                loss3 = e_loss / (self.iter_size * self.config.batch_size)
                loss_e += loss3.item()

                loss = loss1 + loss2 + loss3
                loss_t += loss.item()
                # loss.backward()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    print('Epoch: [%2d/%2d], Iter: [%5d/%5d]  ||  FLoss : %10.4f  ||  SLoss : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, loss_f, loss_t))
                    print('Learning rate: ' + str(self.lr))
                    if i != iter_num:
                        loss_f, loss_s, loss_e, loss_t = 0, 0, 0, 0

            if (epoch >= self.config.epoch * 0.5) and ((epoch + 1) % self.config.epoch_save == 0):
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            self.scheduler.step()
            self.lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss_f]), win='train', update='append' if epoch > 1 else None, opts={'title': 'Train Loss'})

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

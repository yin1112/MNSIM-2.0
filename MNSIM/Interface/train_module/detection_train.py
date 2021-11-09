#-*-coding:utf-8-*-
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.voc2007.tools import gt_creator
from tensorboardX import SummaryWriter
from model.yolo import myNet
from config import cfg
from utils.voc2007.vocapi_evaluator import VOCAPIEvaluator
from datatool.voc2007.voc0712 import BaseTransform
tensorboard_writer = None


def train_net(net, train_loader, test_loader, device, prefix):
    global tensorboard_writer
    tensorboard_writer = SummaryWriter(log_dir = os.path.join(os.path.dirname(__file__), f'runs/{prefix}'))
    # set net on gpu
    net.to(device)
    model = myNet(net, device, input_size=cfg.voc.TRAIN_SIZE, num_classes=cfg.voc.num_classes, trainable=True)
    evaluator = VOCAPIEvaluator(
                            dataset = test_loader , 
                            data_root= cfg.voc.root,
                            img_size=cfg.voc.VAL_SIZE,
                            device=device,
                            transform=BaseTransform(cfg.voc.VAL_SIZE),
                            labelmap=cfg.voc.root
                            )
    # loss and optimizer
    # scale's lr and weight_decay set to 0
    optimizer = optim.SGD([{'params': model.parameters(), 'lr': cfg.train.lr, 'weight_decay': cfg.train.WEIGHT_DECAY}], momentum = cfg.train.MOMENTUM)
    # optimizer = optim.SGD(net.parameters(), lr = lr, weight_decay = WEIGHT_DECAY, momentum = MOMENTUM)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = cfg.train.MILESTONES, gamma = cfg.train.GAMMA)

    # test init
    # eval_net(net, test_loader, 0, device)
    # epochs
    for epoch in range(cfg.train.EPOCHS):
        # train
        model.train()
        scheduler.step()
        for i, (images, targets) in enumerate(train_loader):
            model.zero_grad()
            images = images.to(device)
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=cfg.voc.TRAIN_SIZE, stride=model.stride, label_lists=targets)
            targets = torch.tensor(targets).float().to(device)
            
            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, cfg.MODE, target=targets)
            
            total_loss.backward()
            optimizer.step()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'epoch {epoch+1:3d}, {i:3d}|{len(train_loader):3d}, lr: {lr:.6f}, Loss: obj: {conf_loss:.6f}\
 , cls_loss: {cls_loss:.6f}, txtytwth_loss: {txtytwth_loss:.6f}, loss: {total_loss.item():2.6f}', end = '\r')
            tensorboard_writer.add_scalars('train_loss', {'train_loss': total_loss.item()}, epoch * len(train_loader) + i)
        eval_net(model,evaluator)
        torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), f'zoo/{prefix}_params.pth'))

def eval_net(model, evaluator ):
    # set net on gpu
    model.trainable = False
    model.set_grid(cfg.voc.VAL_SIZE)
    model.eval()

    evaluator.evaluate(model,cfg.MODE)

    # convert to training mode.
    model.trainable = True
    model.set_grid(cfg.voc.TRAIN_SIZE)
    model.train()


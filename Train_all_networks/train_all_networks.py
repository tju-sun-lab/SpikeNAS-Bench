import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
import numpy as np
import random
import gc
from model_snn import SNASNet
from utils import data_transforms, get_cntmat, get_cntmat_3, Get_CIFAR10_train_val, get_module_paramerters, TinyImageNet
from spikingjelly.clock_driven.monitor import Monitor
# from searchcells.monitor_nas import Monitor

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

batch_size = 64


def main():
    args = config.get_args()

    # define dataset
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        train_val_set = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                                 download=True, transform=train_transform)
        num_train_val = len(train_val_set)
        indices = list(range(num_train_val))
        split = int(num_train_val / 2)

        train_loader = torch.utils.data.DataLoader(train_val_set, batch_size=args.batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                   pin_memory=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(train_val_set, batch_size=args.batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train_val]),
                                                   pin_memory=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                               download=True, transform=valid_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=1)
    elif args.dataset == 'cifar100':
        train_val_set = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True,
                                                 download=True, transform=train_transform)
        num_train_val = len(train_val_set)
        indices = list(range(num_train_val))
        split = int(num_train_val / 2)

        train_loader = torch.utils.data.DataLoader(train_val_set, batch_size=args.batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                   pin_memory=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(train_val_set, batch_size=args.batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train_val]),
                                                   pin_memory=True, num_workers=1)
        testset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False,
                                               download=True, transform=valid_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=1)

    elif args.dataset == 'tinyimagenet':

        dataset_train = TinyImageNet(os.path.join(args.data_dir, 'tiny-imagenet-200'), train=True, transform=train_transform)
        dataset_val = TinyImageNet(os.path.join(args.data_dir, 'tiny-imagenet-200'), train=False, transform=train_transform)

        num_train_test = len(dataset_train)
        indices = list(range(num_train_test))
        split = int(num_train_test / 2)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                   pin_memory=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=1)


        test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train_test]),
                                                   pin_memory=True, num_workers=1)



    qualified_mat = 0
    module_name = ['cell1.cell_architecture.0.0', 'cell1.cell_architecture.1.0', 'cell1.cell_architecture.2.0', 'cell1.cell_architecture.3.0', 'cell1.cell_architecture.4.0', 'cell1.cell_architecture.5.0',
                   'cell2.cell_architecture.0.0', 'cell2.cell_architecture.1.0', 'cell2.cell_architecture.2.0', 'cell2.cell_architecture.3.0', 'cell2.cell_architecture.4.0', 'cell2.cell_architecture.5.0',
                   'downconv1.1', 'last_act.1', 'classifier.1']



    for atc_id in range(15624):
        # atc_id = int(atc_id)
        cnt_mat = get_cntmat_3(atc_id)
        qualified_mat += 1
        print('-' * 7, "current_neuroncell", '-' * 7)
        print('architecture id:', atc_id)
        print('cnt_mat:\n', cnt_mat)
        print('-' * 30)

        # Reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(atc_id)
        np.random.seed(atc_id)
        torch.manual_seed(atc_id)
        torch.cuda.manual_seed(atc_id)

        # Master table storing training information
        hist = list()
        # Table storing the firing rate of each layer
        hist_firingrate = list()
        model = SNASNet(args, cnt_mat).to(args.device)
        model_parameters = get_module_paramerters(model)

        # Initialize SNN Monitor
        monitor = Monitor(model, device=args.device, backend='torch')
        criterion = nn.CrossEntropyLoss().to(args.device)

        if args.savemodel_pth is not None:
            print(torch.load(args.savemodel_pth).keys())
            model.load_state_dict(torch.load(args.savemodel_pth)['state_dict'])
            print('test only...')
            validate(args, 0, val_loader, model, criterion, [], [])


        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        else:
            print("will be added...")
            exit()

        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.5),
                                                                                    int(args.epochs * 0.75)],
                                                             gamma=0.1)
        elif args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs),
                                                                   eta_min=args.learning_rate * 0.01)
        else:
            print("will be added...")
            exit()

        start = time.time()
        for epoch in range(args.epochs):
            hist_epoch = []
            train(args, epoch, train_loader, model, criterion, optimizer, scheduler, hist_epoch)
            scheduler.step()
            validate(args, epoch, val_loader, model, criterion, hist, hist_epoch)

        test_acc = test(args, test_loader, model, criterion, monitor, hist_firingrate, module_name)
        hist.append(test_acc)
        hist.append(hist_firingrate)
        train_time = utils.time_record(start)
        utils.save_checkpoint(args, {'state_dict': model.state_dict(), }, atc_id, hist, model_parameters, train_time)
        del hist, hist_firingrate, hist_epoch, model, monitor, criterion, optimizer, scheduler
        gc.collect()


def train(args, epoch, train_data, model, criterion, optimizer, scheduler, hist_epoch):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    if (epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_last_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        train_loss += loss.item()
    hist_epoch.append(epoch)
    hist_epoch.append(top1.avg)
    hist_epoch.append(top5.avg)
    hist_epoch.append((train_loss / len(train_data)))
    print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % top1.avg)


def validate(args, epoch, val_data, model, criterion, hist, hist_epoch):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()
    val_top5 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            val_top5.update(prec5.item(), n)
        hist_epoch.append(val_top1.avg)
        hist_epoch.append(val_top5.avg)
        hist_epoch.append((val_loss / len(val_data)))
        hist.append(hist_epoch)
        print('[Val_Accuracy epoch:%d] val_acc:%f  top5:%f'
              % (epoch + 1, val_top1.avg, val_top5.avg))
        return val_top1.avg

def test(args, test_data, model, criterion, monitor, hist, module_name):
    model.eval()
    monitor.enable()
    test_loss = 0.0
    test_top1 = utils.AvgrageMeter()
    test_top5 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(test_data):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            test_top1.update(prec1.item(), n)
            test_top5.update(prec5.item(), n)
        print('[Test_Accuracy] val_acc:%f'
              % (test_top1.avg))

    for name in module_name:
        if name in monitor.module_dict.keys():
            avg_firing_rate = monitor.get_avg_firing_rate(all=False, module_name=name)
            hist.append(avg_firing_rate)#.cpu().float().item()
        else:
            hist.append(-1)
    avg_firing_rate_all = monitor.get_avg_firing_rate(all=True)
    hist.append(avg_firing_rate_all)#.cpu().float().item()
    monitor.reset()
    return test_top1.avg, test_top5.avg


if __name__ == '__main__':
    main()

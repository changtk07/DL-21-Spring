# SP21 DP Team 05
import os
import sys
import time
import argparse
import shutil

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.append('.')

from custom.dataloader import CustomDataset
from custom.dataloaderP import CustomDatasetPlus
import moco.utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SP21DP Final')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names)
parser.add_argument('-w', '--workers', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--checkpoint', type=str, metavar='PATH', required=True,
                    help='path to save checkpoints')
parser.add_argument('--cos', action='store_true')

def main():
    args = parser.parse_args()

    best_acc1 = 0

    # create model
    print(f'== creating model: {args.arch}')
    model = models.__dict__[args.arch](num_classes=800)

    # freeze all layers but the last fc
    #for name, param in model.named_parameters():
    #    if name not in ['fc.weight', 'fc.bias']:
    #        param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # laod from freezed
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print(f'== loading checkpoint: {args.pretrained}')
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            state_dict = checkpoint['state_dict']
            args.start_epoch = 0
            model.load_state_dict(state_dict, strict=True)

            print(f'== loaded pre-trained model: {args.pretrained}')
        else:
            print(f'== no checkpoint found at: {args.pretrained}')
    
    model.cuda()

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    #assert len(parameters) == 2
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # resume from a checkpoint if specify
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'== loading checkpoint: {args.resume}')

            checkpoint = torch.load(args.resume,
                map_location=torch.device('cuda:0'))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1'].cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('== loaded checkpoint: {} (epoch {})'
                .format(args.resume, checkpoint['epoch']))
        else:
            print(f'== no checkpoint found at: {args.resume}')

    # this can speed up train speed
    cudnn.benchmark = True

    # loading data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CustomDatasetPlus(root='/dataset',
                                      addition='/home/tc3149/addition',
                                      transform=train_transform)
    val_dataset = CustomDataset(root='/dataset', split='val',
                                transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('== start training')

    for epoch in range(args.start_epoch, args.epochs):
        moco.utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(model, val_loader, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save checkpoint
        torch.save(
            {'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_acc1': best_acc1,
            },
            args.checkpoint + 'checkpoint_{:04d}.pth.tar'.format(epoch)
        )
        if is_best:
            shutil.copyfile(
                args.checkpoint + 'checkpoint_{:04d}.pth.tar'.format(epoch),
                args.checkpoint + 'best_{:04d}.pth.tar'.format(epoch)
            )

def train(model, train_loader, criterion, optimizer, epoch, args):
    batch_time = moco.utils.AverageMeter('Time', ':6.3f')
    data_time = moco.utils.AverageMeter('Data', ':6.3f')
    losses = moco.utils.AverageMeter('Loss', ':.4e')
    top1 = moco.utils.AverageMeter('Acc@1', ':6.2f')
    top5 = moco.utils.AverageMeter('Acc@5', ':6.2f')
    progress = moco.utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = moco.utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(model, val_loader, criterion, args):
    batch_time = moco.utils.AverageMeter('Time', ':6.3f')
    losses = moco.utils.AverageMeter('Loss', ':.4e')
    top1 = moco.utils.AverageMeter('Acc@1', ':6.2f')
    top5 = moco.utils.AverageMeter('Acc@5', ':6.2f')
    progress = moco.utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader, 1):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = moco.utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return top1.avg

if __name__ == '__main__':
    main()

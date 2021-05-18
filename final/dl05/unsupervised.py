# SP21 DP Team 05
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.append('.')

from custom.dataloader import CustomDataset
import moco.loader
import moco.builder
import moco.utils

# argument removed: seed, aug_plus, all distributed args
# argument added: checkpoint

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SP21DP Final')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names)
parser.add_argument('-w', '--workers', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', type=str, metavar='PATH', required=True,
                    help='path to save checkpoints')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int)
parser.add_argument('--moco-k', default=65536, type=int)
parser.add_argument('--moco-m', default=0.999, type=float)
parser.add_argument('--moco-t', default=0.07, type=float)
parser.add_argument('--mlp', action='store_true')
parser.add_argument('--cos', action='store_true')

def main():
    args = parser.parse_args()

    print('== checkpoints dir: {}'.format(args.checkpoint))

    # create checkpoint dir
    os.makedirs(args.checkpoint, exist_ok=True)

    # create models
    print(f'== creating model: {args.arch}')
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    model.cuda()

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # resume from a checkpoint if specify
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'== loading checkpoint: {args.resume}')

            checkpoint = torch.load(args.resume,
                map_location=torch.device('cuda:0'))

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('== loaded checkpoint: {} (epoch {})'
                    .format(args.resume, checkpoint['epoch']))
        else:
            print(f'== no checkpoint found at {args.resume}')

    # this can speed up train speed
    cudnn.benchmark = True

    # loading data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.RandomApply([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation))

    train_dataset = CustomDataset(root='/dataset', split='unlabeled',
                                  transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    print('== start training')

    for epoch in range(args.start_epoch, args.epochs):
        moco.utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch, args)

        # save checkpoint
        torch.save(
            {'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            },
            args.checkpoint + 'checkpoint_{:04d}.pth.tar'.format(epoch)
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
    for i, (images, _) in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = moco.utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

if __name__ == '__main__':
    main()

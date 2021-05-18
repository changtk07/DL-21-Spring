import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pandas as pd
import numpy as np

sys.path.append('.')

from custom.dataloader import CustomDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SP21DP Final')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names)
parser.add_argument('-w', '--workers', default=8, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('-p', '--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', type=str, metavar='PATH', required=True)

def validate(model, val_loader, args):
    model.eval()

    total = torch.empty(0).cuda()

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.cuda(non_blocking=True)

            # compute output
            output = model(images)
            prob = nn.Softmax(dim=1)(output)

            total = torch.cat((total, prob), 0)
    
    x_np = total.cpu().numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv(args.checkpoint + 'prob.csv')

def main():
    args = parser.parse_args()

    # create model
    print(f'== creating model: {args.arch}')
    model = models.__dict__[args.arch](num_classes=800)

    model.cuda()

    # resume from a checkpoint if specify
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'== loading checkpoint: {args.resume}')

            checkpoint = torch.load(args.resume,
                map_location=torch.device('cuda:0'))
            model.load_state_dict(checkpoint['state_dict'])
            print('== loaded checkpoint: {} (epoch {})'
                .format(args.resume, checkpoint['epoch']))
        else:
            print(f'== no checkpoint found at: {args.resume}')

    # loading data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = CustomDataset(root='/dataset', split='unlabeled',
                                  transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('== start testing')
    validate(model, val_loader, args)

if __name__ == '__main__':
    main()

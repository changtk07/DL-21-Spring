# SP21 DP Team 05
import sys
import csv

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.append('.')

from custom.dataloader import CustomDataset
from custom.dataloaderP import CustomDatasetPlus

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_dataset = CustomDataset(root='/dataset', split='train',
                              transform=transform)

unlabeled_dataset = CustomDataset(root='/dataset', split='unlabeled',
                                  transform=transform)

plus_dataset = CustomDatasetPlus(root='/dataset', 
                                 addition='/home/tc3149/addition',
                                 transform=transform)

#train_loader = torch.utils.data.DataLoader(
#        train_dataset,
#        batch_size=1,
#        shuffle=False,
#        num_workers=8
#)

#unlabeled_loader = torch.utils.data.DataLoader(
#        unlabeled_dataset,
#        batch_size=1,
#        shuffle=False,
#        num_workers=8
#)

#plus_loader = torch.utils.data.DataLoader(
#        plus_dataset,
#        batch_size=1,
#        shuffle=True,
#        num_workers=8
#)

with open('/home/tc3149/addition/request_05.csv') as f:
    reader = csv.reader(f)
    request_imgs = [ int(it[0][:-4]) for it in reader ]

request_labels = torch.load('/home/tc3149/addition/label_05.pt', map_location='cpu')

for i in range(len(plus_dataset)):
    pimg, plabel = plus_dataset[i]
    if i < 25600:
        timg, tlabel = train_dataset[i]
        assert torch.equal(pimg, timg)
        assert torch.equal(plabel, tlabel)
    else:
        idx = request_imgs[i-25600]
        uimg, _ = unlabeled_dataset[idx]
        assert torch.equal(pimg, uimg)

print('all test passed')


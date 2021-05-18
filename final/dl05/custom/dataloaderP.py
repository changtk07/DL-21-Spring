# SP21 DP Team 05
import os
import csv

from PIL import Image

import torch

class CustomDatasetPlus(torch.utils.data.Dataset):
    def __init__(self, root, addition, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            addition: Location of the additional csv file containing 
            image indices and requested label file.
            transform: the transform you want to applied to the images.
        """
        self.transform = transform

        self.train_dir = os.path.join(root, 'train')
        self.unlabeled_dir = os.path.join(root, 'unlabeled')
        train_label_path = os.path.join(root, 'train_label_tensor.pt')
        request_label_path = os.path.join(addition, 'label_05.pt')

        with open(os.path.join(addition, 'request_05.csv')) as f:
            reader = csv.reader(f)
            self.request_imgs = [ int(it[0][:-4]) for it in reader ]

        self.train_labels = torch.load(train_label_path)
        self.request_labels = torch.load(request_label_path)

        self.num_images = len(os.listdir(self.train_dir)) + len(self.request_imgs)
        assert self.num_images == 38400

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if idx < 25600:
            with open(os.path.join(self.train_dir, f"{idx}.png"), 'rb') as f:
                img = Image.open(f).convert('RGB')

            return self.transform(img), self.train_labels[idx]
        else:
            idx -= 25600
            with open(os.path.join(self.unlabeled_dir, f"{self.request_imgs[idx]}.png"), 'rb') as f:
                img = Image.open(f).convert('RGB')

            return self.transform(img), self.request_labels[idx]

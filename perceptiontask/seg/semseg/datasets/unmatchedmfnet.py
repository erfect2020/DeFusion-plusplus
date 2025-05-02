import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation


class UNMATCHEDMFNet(Dataset):
    """
    num_classes: 9
    """
    CLASSES = ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
    PALETTE = torch.tensor(
        [[64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128],
         [192, 64, 0]])

    def __init__(self, root: str = 'data/MFNet', split: str = 'train', transform=None, modals=['img', 'thermal'],
                 case=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        rgb = os.path.join(*[self.root, 'rgb', item_name + '.png'])
        x1 = os.path.join(*[self.root, 'ther', item_name + '.png'])
        lbl_path = os.path.join(*[self.root, 'labels', item_name + '.png'])
        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_img(x1)
        print("read image", sample['img'].max())
        label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
        sample['mask'] = label

        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        # print("image scale", sample['img'].max())
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()

        sample = [sample[k] for k in self.modals]
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                file_name = file_name.split(' ')[0]
            file_names.append(file_name)
        return file_names


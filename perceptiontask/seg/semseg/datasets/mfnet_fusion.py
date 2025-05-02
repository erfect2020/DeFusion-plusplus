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


class MFNetFusion(Dataset):
    """
    num_classes: 9
    """
    CLASSES = ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
    PALETTE = torch.tensor([[64,0,128],[64,64,0],[0,128,192],[0,0,192],[128,128,0],[64,64,128],[192,128,128],[192,64,0], [161, 14, 121]])
    # 161, 14, 121 [88, 72, 96]
    def __init__(self, root: str = 'data/MFNet', split: str = 'train', transform = None, modals = ['img'], case = None) -> None:
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
        item_name = self.files[index]
        # print("item name", item_name)
        try:
            x1, lbl_path = item_name
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        sample = {}
        # sample['img'] = io.read_image(x1)[:3, ...]
        sample['img'] = self._open_img(x1)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        sample['mask'] = label
        # print("label max", label.max())

        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        # print("image scale", sample['img'].max())
        label = self.encode(label.squeeze().numpy()).long()
        # print("lable max after encode", label.max())
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
        source = os.path.join(self.root, 'test') if split_name == 'val' else os.path.join(self.root, 'train')

        ir_source = os.path.join(source, 'MetaFusion')
        lab_source = os.path.join(source, 'Segmentation_labels')
        ir_file_names = [(os.path.join(ir_source, ir), os.path.join(lab_source, label)) for ir, label in zip(sorted(os.listdir(ir_source)), sorted(os.listdir(lab_source)))]
        return ir_file_names



if __name__ == '__main__':
    traintransform = get_train_augmentation((480, 640), seg_fill=255)

    trainset = MFNetFusion(transform=traintransform)
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))

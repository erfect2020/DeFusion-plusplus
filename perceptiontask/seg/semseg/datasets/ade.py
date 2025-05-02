from torch.utils.data import Dataset
import os
import json
import torch
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, RandomCrop, ColorJitter
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from PIL import Image
import random
from torch import Tensor
from torchvision import io




class AdeDataset(Dataset):
    CLASSES = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                    'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                    'person', 'earth', 'door', 'table', 'mountain', 'plant',
                    'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                    'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                    'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                    'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                    'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                    'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                    'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                    'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                    'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                    'chandelier', 'awning', 'streetlight', 'booth',
                    'television receiver', 'airplane', 'dirt track', 'apparel',
                    'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                    'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                    'conveyer belt', 'canopy', 'washer', 'plaything',
                    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                    'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                    'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                    'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                    'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                    'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                    'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                    'clock', 'flag']
    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                    [102, 255, 0], [92, 0, 255]]
    def __init__(self, root: str = 'data/MFNt', split: str = 'train', transform = None, modals = ['img', 'thermal'], case = None):
        super(AdeDataset, self).__init__()

        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.root = root
        self.transform = transform

        self.epoch_num = 0
        self.same_radio = 0.2
        self.noise_radio = 0.2
        self.grid_num1 = 14 #14
        self.rand_dist1 = Bernoulli(probs=torch.ones(self.grid_num1 ** 2) * 0.99)
        self.grid_num2 = 7 #56 # 28
        self.rand_dist2 = Bernoulli(probs=torch.ones(self.grid_num2 ** 2) * 0.95)

        self.files = self._get_file_names(split)

        if not self.files:
            raise Exception(f"No images found in {self.path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self):
        return len(self.files)

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source_img = os.path.join(self.root, 'images/validation') if split_name == 'val' else os.path.join(self.root, 'images/training')
        source_lab = os.path.join(self.root, 'annotations/validation') if split_name == 'val' else os.path.join(self.root,
                                                                                                           'annotations/training')
        source_imgs = sorted(os.listdir(source_img))
        source_labs = sorted(os.listdir(source_lab))

        ir_file_names = [( os.path.join(source_img, vi), os.path.join(source_lab, label)) for vi, label in zip(source_imgs, source_labs)]
        return ir_file_names

    def reassign_mask(self, radio):
        self.rand_dist1 = Bernoulli(probs=torch.ones(self.grid_num1 ** 2) * radio)
        self.rand_dist2 = Bernoulli(probs=torch.ones(self.grid_num2 ** 2) * radio)

    def __getitem__(self, index):
        rgb, lbl_path = self.files[index]

        sample = {}
        try:
            sample['img'] = io.read_image(rgb)[:3, ...]
            sample['thermal'] = io.read_image(rgb)[:3, ...]
            label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
            sample['mask'] = label
        except:
            try:
                sample['img'] = io.read_image('/home/lpw/Documents/fusion-dataset/ADEChallengeData2016/images/training/ADE_train_00000001.jpg')[:3, ...]
                sample['thermal'] = io.read_image(
                    '/home/lpw/Documents/fusion-dataset/ADEChallengeData2016/images/training/ADE_train_00000001.jpg')[
                                :3, ...]
                sample['mask'] = io.read_image(
                    '/home/lpw/Documents/fusion-dataset/ADEChallengeData2016/annotations/training/ADE_train_00000001.png')[0, ...].unsqueeze(0)
            except:
                print("error", rgb, lbl_path)

        # print("shape", sample['img'].shape)
        if sample['img'].shape[0] == 1:
            sample['img'] = sample['img'].repeat(3, 1, 1)
        if sample['thermal'].shape[0] == 1:
            sample['thermal'] = sample['thermal'].repeat(3, 1, 1)

        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()

        o_img = sample['img']
        u_img = sample['thermal']
        self.image_size =(480, 480)
        # print("images size", self.image_size)

        self.grid_num1, self.grid_num2 = 10, 30
        self.reassign_mask(0.1 + torch.rand(1).item()/1.12)
        grid1 = F.interpolate(self.rand_dist1.sample().reshape(1, 1, self.grid_num1, self.grid_num1),
                              size=self.image_size, mode='nearest').squeeze()
        grid1 *= F.interpolate(self.rand_dist2.sample().reshape(1, 1, self.grid_num2, self.grid_num2),
                              size=self.image_size, mode='nearest').squeeze()
        grid2 = F.interpolate(self.rand_dist1.sample().reshape(1, 1, self.grid_num1, self.grid_num1),
                              size=self.image_size, mode='nearest').squeeze()
        grid2 *= F.interpolate(self.rand_dist2.sample().reshape(1, 1, self.grid_num2, self.grid_num2),
                              size=self.image_size, mode='nearest').squeeze()

        sample_rand = torch.rand(1).item()
        if sample_rand < 0.5:
            grid3 = F.interpolate(self.rand_dist1.sample().reshape(1, 1, self.grid_num1, self.grid_num1),
                               size=self.image_size, mode='nearest').squeeze()
        else:
            grid3 = F.interpolate(self.rand_dist2.sample().reshape(1, 1, self.grid_num2, self.grid_num2),
                              size=self.image_size, mode='nearest').squeeze()

        none_grid = ((grid1 == 0.0) & (grid2 == 0.0)).float()
        none_grid1 = grid3 * none_grid
        none_grid2 = (1 - grid3) * none_grid
        grid1 += none_grid1
        grid2 += none_grid2

        if torch.rand(1).item() < 0.5:
            mask1, mask2 = grid1, grid2
        else:
            mask1, mask2 = grid2, grid1

        o_rand = torch.randn_like(o_img).abs().clamp(0,1)
        u_rand = torch.randn_like(u_img).abs().clamp(0,1)
        # print("mask shape", mask2.shape, mask1.shape, o_img.shape, u_img.shape)
        o_img = o_img * mask1 + o_rand * (1.0 - mask1) #* torch.rand(1).item()
        u_img = u_img * mask2 + u_rand * (1.0 - mask2) #* torch.rand(1).item()

        sample = [o_img, u_img]
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
        # label = label.where(label == 150, torch.tensor(255), label)
        # print("label", label.min(), label.max(), len(self.CLASSES))
        return sample, label

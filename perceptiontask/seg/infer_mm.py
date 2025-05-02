import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import torchvision.transforms.functional as TF 
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.augmentations_mm import get_val_augmentation
from semseg.utils.visualize import draw_text
import glob
import os
from PIL import Image, ImageDraw, ImageFont


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        self.model.init_pretrained(cfg['MODEL']['PRETRAINED'])
        msg = self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        # self.tf_pipeline_img = T.Compose([
        #     T.Resize(self.size),
        #     T.Lambda(lambda x: x / 255),
        #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     T.Lambda(lambda x: x.unsqueeze(0))
        # ])
        # self.tf_pipeline_modal = T.Compose([
        #     T.Resize(self.size),
        #     T.Lambda(lambda x: x / 255),
        #     T.Lambda(lambda x: x.unsqueeze(0))
        # ])

        self.tf_pipeline_all = get_val_augmentation(self.size)

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        return pil_image

    def labelpostprocess(self, orig_img: Tensor, label: Tensor, overlay: bool) -> Tensor:
        seg_map = label.cpu().to(int)

        print(seg_map.shape, self.palette.shape, seg_map)
        seg_image = self.palette[seg_map].squeeze()
        # seg_image = seg_map
        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        return pil_image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
    
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

    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        if cfg['DATASET']['NAME'] == 'MFNetFusion':
            x1 = img_fname.replace('/img', '/ir').replace('_rgb', '_depth')
            x2 = img_fname.replace('/img', '/vis').replace('_rgb', '_lidar')
            lbl_path = img_fname.replace('/fusion', '/Segmentation_labels')
        elif cfg['DATASET']['NAME'] == 'MFNet':
            x2 = os.path.join(img_fname.replace('vi', 'ir'))
            # x2 = x2.replace('.png', '_color.png')
            lbl_path = img_fname.replace('/vi', '/Segmentation_labels')

        image = io.read_image(img_fname)[:3, ...]
        # image = self._open_img(img_fname)
        sample = {}
        sample['img'] = image
        # --- modals
        x2 = self._open_img(x2)
        sample['thermal'] = x2
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        sample['mask'] = label
        sample = self.tf_pipeline_all(sample)

        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        # sample = [sample[k].unsqueeze(0).cuda() for k in ['img', 'thermal']]
        sample = [sample[k].unsqueeze(0).cuda() for k in ['img']]

        seg_map = self.model_forward(sample)
        seg_map = self.postprocess(image, seg_map, overlay)
        # seg_map = self.labelpostprocess(image, label, overlay)
        return seg_map



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    modals = cfg['DATASET']['MODALS']

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['SAVE_DIR']) / 'test_results' / (cfg['DATASET']['NAME']+'_'+cfg['MODEL']['BACKBONE']+'_'+modals_name)
    
    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"{str(test_file.stem)}.png")
    else:
        if cfg['DATASET']['NAME'] == 'MFNetFusion':
            files = sorted(glob.glob(os.path.join(*[str(test_file), '*.jpg']))) # --- Deliver
        elif cfg['DATASET']['NAME'] == 'MFNet':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'vi', '*.png'])))
        else:
            raise NotImplementedError()

        for file in files:
            print(file)
            segmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            save_path = os.path.join(str(save_dir),file)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            segmap.save(save_path)

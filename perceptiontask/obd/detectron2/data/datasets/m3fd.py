# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_m3fd_instances", "register_m3fd"]


# fmt: off
CLASS_NAMES = ('People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck')
# fmt: on


def load_m3fd_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        vis_file = os.path.join(dirname, "Vis", fileid + ".png")
        ir_file = os.path.join(dirname, "Ir", fileid + ".png")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            'vis_name': vis_file,
            'ir_name': ir_file,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

M3FD_CATEGORY_IMAGE_COUNT = [{'id': 1, 'image_count': 9320}, {'id': 2, 'image_count': 14739}, {'id': 3, 'image_count': 549}, {'id': 4, 'image_count': 1982},  {'id': 5, 'image_count': 424}, {'id': 6, 'image_count': 816}]  # noqa

def register_m3fd(name, dirname, split, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_m3fd_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, split=split, class_image_count=M3FD_CATEGORY_IMAGE_COUNT
    )

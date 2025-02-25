<div align="center">
  
# Fusion from Decomposition: A Self-Supervised Approach for Image Fusion and Beyond

[Pengwei Liang](https://scholar.google.com/citations?user=54Ci0_0AAAAJ&hl=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ), [Qing Ma](https://scholar.google.com/citations?user=x6QQGQkAAAAJ&hl=en), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ)

Harbin Institute of Technology, Harbin 150001, China. Electronic Information School, Wuhan University, Wuhan 430072, China.
</div>

## [Paper](https://arxiv.org/abs/2410.12274)

> Image fusion is famous as an alternative solution to generate one high-quality image from multiple images in addition to image restoration from a single degraded image. The essence of image fusion is to integrate complementary information from source images. Existing fusion methods struggle with generalization across various tasks and often require labor-intensive designs, in which it is difficult to identify and extract useful information from source images due to the diverse requirements of each fusion task. Additionally, these methods develop highly specialized features for different downstream applications, hindering the adaptation to new and diverse downstream tasks. To address these limitations, we introduce DeFusion++, a novel framework that leverages self-supervised learning (SSL) to enhance the versatility of feature representation for different image fusion tasks. DeFusion++ captures the image fusion task-friendly representations from large-scale data in a self-supervised way, overcoming the constraints of limited fusion datasets. Specifically, we introduce two innovative pretext tasks: common and unique decomposition (CUD) and masked feature modeling (MFM). CUD decomposes source images into abstract common and unique components, while MFM refines these components into robust fused features. Jointly training of these tasks enables DeFusion++ to produce adaptable representations that can effectively extract useful information from various source images, regardless of the fusion task. The resulting fused representations are also highly adaptable for a wide range of downstream tasks, including image segmentation and object detection. DeFusion++ stands out by producing versatile fused representations that can enhance both the quality of image fusion and the effectiveness of downstream high-level vision tasks, simplifying the process with the elegant fusion framework. We evaluate our approach across three publicly available fusion tasks involving infrared and visible fusion, multi-focus fusion, and multi-exposure fusion, as well as on downstream tasks. The results, both qualitative and quantitative, highlight the versatility and effectiveness of DeFusion++.

## Virtual Environment

```python
conda create -n DeFusion++ python=3.9
conda install -r requirement.txt
```

## Testing

1. Downloading the [pre-trained model]() and placing them in **./pretrained** .
2. Run the following script for fusion testing:

```python
## for multi-modal image fusion
python test.py -opt option/test/MIVF_TransformerTest_Dataset.yaml
## for MEF image fusion
python test.py -opt option/test/MEF_TransformerTest_Dataset.yaml
## for MFF image fusion
python test.py -opt option/test/MFF_TransformerTest_Dataset.yaml
```
3. Change the data loader in `test.py` to load test datasets in different formats.



## Training

### Training CUD

1. Download [COCO](https://github.com/cocodataset/cocoapi): https://cocodataset.org/
2. Put your training images into any folder and modify the parameter `dataroot` in `option/train/SelfTrained_SDatasetFast.yaml` to update the path.
3. Run the following script for training the DeFusion++:

```python
## for training DeFusion++ on the COCO dataset
python selftrain.py -opt option/train/SelfTrained_SDatasetFast.yaml
```

### Training MCUD

1. Download our pretrained MAE model on [FLIR]() and pretrained MAE model on [ImageNet]().
2. Modify the path of pretrained model in `models/MM.py`
3. Put the training images (COCO/VIF dataset) into any folder and modify the parameter `noise_dataroot, irvis_dataroot` in `option/train/SelfTrained_MDatasetFast.yaml` to update the path.

```python
python selftrain_multimodal.py -opt option/train/SelfTrained_MDatasetFast.yaml
```

## Bib 
If this repo help you, please cite us:
```
@article{liang2025fusion,
  title={Fusion from decomposition: A self-supervised approach for image fusion and beyond},
  author={Liang, Pengwei and Jiang, Junjun and Ma, Qing and Liu, Xianming and Ma, Jiayi},
  journal={arXiv preprint arXiv:2410.12274},
  year={2024}
}
```

# -*- coding: utf-8 -*-
from .resnet import resnet
from .yolov5 import YOLOv5
from .swin_transformer import swin_transformer as swin
from .vit import vit 
from .vgg import vgg

__all__ = ['build_backbone']

support_backbone = ['resnet', 'YOLOv5', 'swin', 'vgg', 'vit']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone

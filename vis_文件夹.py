import os
import sys
import cv2
import torch
import numpy as np
# import lightning as L
import matplotlib.pyplot as plt
# from lightning.fabric.fabric import _FabricOptimizer
# from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks


import argparse
import os
import glob

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
# from mmcv.runner import load_checkpoint

def visualize(cfg):
    cfg = utils.load_config(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/demo0610_allshadow_l.yaml')
    parser.add_argument('--model', default='save/_demo0610_allshadow_l/model_epoch_best.pth')
    parser.add_argument('--input_folder', default='load/shadow_steel/test/images')
    parser.add_argument('--output_folder', default='out/')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 模型加载
    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    with open('layer_names.txt', 'w') as f:
        for name, _ in model.named_modules():
            f.write(name + '\n')

    print("Layer names have been saved to 'layer_names.txt'.")
    
    model.eval()

    # 获取文件夹中所有图片的路径
    image_paths = glob.glob(os.path.join(args.input_folder, '*.png'))

    for image_path in image_paths:
        # 读取图片
        inp = cv2.imread(image_path)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (1024, 1024), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(inp).float() / 255.0
        inp = inp.permute(2, 0, 1).unsqueeze(0).cuda()
         # 推理
        with torch.no_grad():
            pred_mask = model.infer(inp)
            pred_mask = torch.sigmoid(pred_mask)

        # 转换mask为NumPy数组并保存
        mask_to_display = pred_mask[0, 0].cpu().numpy()
        mask_filename = os.path.join(args.output_folder, os.path.basename(image_path))
        cv2.imwrite(mask_filename, mask_to_display * 255)

        print(f'Mask saved to {mask_filename}')

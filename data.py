import torchio as tio
import numpy as np
import torch
from torch.utils.data import Dataset

class LungNoduleDataset(Dataset):
    def __init__(self, data_paths, mode='train', img_size=256):
        self.mode = mode
        self.preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.Resample((1.0, 1.0, 1.0)),  # 统一重采样到1mm各向同性
            tio.CropOrPad((img_size, img_size, 32)),  # 固定深度为32层
            tio.ZNormalization(),  # Z-score标准化
        ])

        self.augment = tio.Compose([
            tio.RandomElasticDeformation(
                num_control_points=4,
                locked_borders=2,
                probability=0.8 if mode=='train' else 0),
            tio.RandomFlip(axes=('LR',), probability=0.5),
            tio.RandomNoise(mean=0, std=0.01, p=0.3),
        ])

        # 加载数据路径 (需根据实际数据格式实现)
        self.samples = self._load_data(data_paths)

    def _load_data(self, data_paths):
        # 实现CT和标注的加载逻辑
        # 返回格式: [{'image': tio.ScalarImage, 'label': tio.LabelMap}, ...]
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        preprocessed = self.preprocess(sample)
        if self.mode == 'train':
            preprocessed = self.augment(preprocessed)

        image = preprocessed['image'].data  # (1, H, W, D)
        label = preprocessed['label'].data.float()  # (1, H, W, D)

        # 转换为2.5D切片处理 (5层堆叠)
        slices = self._extract_25d_slices(image, label)
        return slices

    def _extract_25d_slices(self, volume, label):
        # 提取中心层及相邻±2层形成伪3D块
        depth = volume.shape[-1]
        center = depth // 2
        indices = [max(0, center-2), center, min(depth-1, center+2)]

        image_stack = torch.stack([volume[...,i] for i in indices], dim=-1)  # (1, H, W, 5)
        label_slice = label[..., center]  # (1, H, W)

        return {
            'image': image_stack.squeeze(0).permute(2,0,1),  # (5, H, W)
            'label': label_slice.squeeze(0)  # (H, W)
        }
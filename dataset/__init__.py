import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from augment.mask  import FrequencyMaskGenerator

from dataset.dataset import DGM4_Dataset
from dataset.randaugment import RandomAugment
from augment.mask import *
from utils import *
from augment.augment import *


def create_dataset(config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])    
    
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    
    train_dataset = DGM4_Dataset(config=config, ann_file=config['train_file'], transform=train_transform, max_words=config['max_words'], is_train=True)              
    val_dataset = DGM4_Dataset(config=config, ann_file=config['val_file'], transform=test_transform, max_words=config['max_words'], is_train=False)              
    return train_dataset, val_dataset    

from torchvision import transforms
from PIL import Image
import numpy as np
from augment.mask import FrequencyMaskGenerator
from dataset.dataset import DGM4_Dataset
from dataset.randaugment import RandomAugment
from utils import *
from augment.augment import *

# 定义一个数据增强和遮罩生成的处理类
class TransformWithMask:
    def __init__(self, train_transform, mask_generator):
        self.train_transform = train_transform
        self.mask_generator = mask_generator

    def __call__(self, image):
        # 确保图像是PIL Image格式
        
        if image is None:
            raise ValueError("Input image is None.")

        # 如果是 numpy.ndarray 类型，则转换为 PIL Image
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] not in [1, 3]:  # 确保是一个彩色或灰度图
                raise ValueError(f"Invalid shape for ndarray image: {image.shape}")
                

    # 如果是 PIL.Image 类型，直接返回
        elif isinstance(image, Image.Image):
            pass  # 如果是 PIL.Image 类型，则跳过

    # 如果是其他类型，抛出错误
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected numpy.ndarray or PIL.Image.")
        # print(type(image))
        # 应用训练时的增强变换
        image = self.train_transform(image)
        
        # 应用遮罩生成
        image = self.mask_generator.transform(image)
        return image

# def create_dataset(config):
#     normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

#     # 在这里定义训练时的数据增强操作
#     train_transform = transforms.Compose([
#         RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
#         transforms.ToTensor(),
#         normalize,
#     ])    

#     # 定义验证集的变换
#     test_transform = transforms.Compose([
#         transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
#         transforms.ToTensor(),
#         normalize,
#     ])  

#     mask_generator = FrequencyMaskGenerator(ratio=0.15, band='all')

#     # 使用 TransformWithMask 类，避免使用局部函数
#     train_transform_with_mask = TransformWithMask(train_transform, mask_generator)
    
#     # 定义训练和验证数据集
#     train_dataset = DGM4_Dataset(config=config, ann_file=config['train_file'], transform=train_transform_with_mask, max_words=config['max_words'], is_train=True)              
#     val_dataset = DGM4_Dataset(config=config, ann_file=config['val_file'], transform=test_transform, max_words=config['max_words'], is_train=False)              
    
#     return train_dataset, val_dataset

    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders

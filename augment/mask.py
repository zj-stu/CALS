import torch
import math
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
import numpy as np
import torch.fft as fft
import torch.nn.functional as F

# from dataset import *

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from imageio import imsave


class FrequencyMaskGenerator:
    def __init__(self, ratio: float = 0.3, band: str = 'all') -> None:
        self.ratio = ratio
        self.band = band  # 'low', 'mid', 'high', 'all'

    def transform(self, image: Image.Image) -> Image.Image:
        image_array = np.array(image).astype(np.complex64)#(3,256,256)
        freq_image = np.fft.fftn(image_array, axes=(0, 1))
        # freq_image = freq_image.transpose(1,2,0)
        # print('freq_image:',freq_image.shape)
        _,height, width = image_array.shape

        mask = self._create_balanced_mask(height, width)
        self.masked_freq_image = freq_image * mask  
        masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real

        # print(masked_image_array.shape)
        # print(type(masked_image_array))
        # masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        return masked_image_array



    
    def _create_balanced_mask(self, height, width):
        mask = np.ones((3,height, width), dtype=np.complex64)

        # Determine the region of the frequency domain to mask
        if self.band == 'low':
            y_start, y_end = 0, height // 4
            x_start, x_end = 0, width // 4
        elif self.band == 'mid':
            y_start, y_end = height // 4, 3 * height // 4
            x_start, x_end = width // 4, 3 * width // 4
        elif self.band == 'high':
            y_start, y_end = 3 * height // 4, height
            x_start, x_end = 3 * width // 4, width
        elif self.band == 'all':
            y_start, y_end = 0, height
            x_start, x_end = 0, width
        else:
            raise ValueError(f"Invalid band: {self.band}")

        num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))#np.ceil(...)：使用 NumPy 库的 ceil 函数将上述计算结果向上取整
        mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]#NumPy 库中的 permutation 函数会生成一个随机排列的数组。这里的输入是一个整数，表示生成一个从 0 到 (y_end - y_start) * (x_end - x_start) - 1 的数组，并将其随机打乱
        y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
        x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

        mask[:,y_indices, x_indices] = 0
        return mask


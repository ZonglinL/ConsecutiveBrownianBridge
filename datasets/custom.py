import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import *
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os



@Registers.datasets.register_with_name('Interpolation')
class Interpolation(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        if stage == 'train':
            self.imgs = Vimeo(self.image_size,self.flip,self.to_normal,dataset_config.aug_noise,dataset_config.aug_cut)
        elif stage == 'test':
            if dataset_config.eval == 'UCF':
                self.imgs = UCF(self.image_size,self.flip,self.to_normal)
            elif dataset_config.eval == 'MidB':
                self.imgs = MidB(self.image_size,self.flip,self.to_normal)
            elif dataset_config.eval == 'DAVIS':
                self.imgs = DAVIS(self.image_size,self.flip,self.to_normal)
            elif dataset_config.eval == 'FILM':
                self.imgs = FILM(self.image_size,self.flip,self.to_normal,dataset_config.mode)
        else:
            self.imgs = UCF(self.image_size,self.flip,self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i]
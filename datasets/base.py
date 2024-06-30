from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import os
import torch

import torchvision.transforms.functional as TF


def rand_crop(*args, sz):
    i, j, h, w = transforms.RandomCrop.get_params(args[0], output_size=sz)
    out = []
    for im in args:
        out.append(TF.crop(im, i, j, h, w))
    return out

def cut(img, high,stride):
    coord_1,coord_2 = np.random.randint(low = 0,high = high),np.random.randint(low = 0,high = high)

    img[:,coord_1:coord_1+stride,coord_2:coord_2+stride] = 1 ## cut to be white

    return img




class Vimeo(Dataset):
    ## interpolation dataset for UCF
    def __init__(self, image_size=(256, 256), flip=True, to_normal=True):
        self.image_size = image_size
        self.root = "data/vimeo_triplet"
        f = open(os.path.join(self.root,'tri_trainlist.txt'), "r")
        imlist = f.read()
        self.image_dirs = imlist.split('\n')[:-1]

        self._length = len(self.image_dirs) ## folder of the images
        self.to_normal = to_normal # normalize to -1,1
        self.flip = flip ## if flip or not

    def __len__(self):
        return self._length

    def load_image(self,img_path,transform):
        try:
            image = Image.open(img_path)
        except:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        return image

    def __getitem__(self, index):
       

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        

        img_path_first = os.path.join(self.root,'sequences',self.image_dirs[index],"im1.png") ## y in BBDM
        img_path_second = os.path.join(self.root,'sequences',self.image_dirs[index],"im3.png") ## z in BBDM
        img_path_target = os.path.join(self.root,'sequences',self.image_dirs[index],"im2.png") ## x in BBDM


        x,y,z = self.load_image(img_path_target,transform),self.load_image(img_path_first,transform),self.load_image(img_path_second,transform)

        x,y,z = rand_crop(x,y,z,sz = self.image_size)

        ## horizontal and vertical flip

        horiz_flip = transforms.RandomHorizontalFlip(1.)
        vert_flip = transforms.RandomVerticalFlip(1.) 
        if np.random.rand()<0.5:
            x,y,z = horiz_flip(x),horiz_flip(y),horiz_flip(z)
        if np.random.rand()<0.5:
            x,y,z = vert_flip(x),vert_flip(y),vert_flip(z)
    
        if np.random.rand() < 0.5:
            return x,y,z
        else:
            return x,z,y



class UCF(Dataset):
    ## interpolation dataset for UCF
    def __init__(self, image_size=(256, 256), flip=False, to_normal=True):
        self.image_size = image_size
        self.image_dirs = []
        for root,dirs,files in os.walk('data/UCF'):
            for file in files:
                if 'png' in file or 'jpg' in file:
                    self.image_dirs.append(root)
                    break

        self._length = len(self.image_dirs) ## folder of the images
        self.to_normal = to_normal # # if normalize to [-1, 1] nor not
        self.flip = flip ## if flip or not

    def __len__(self):

        return self._length

    def load_image(self,img_path,transform):
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)
        

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)


        return image

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        img_path_first = os.path.join(self.image_dirs[index],"frame_00.png") ## y in BBDM
        img_path_second = os.path.join(self.image_dirs[index],"frame_02.png") ## z in BBDM
        img_path_target = os.path.join(self.image_dirs[index],"frame_01_gt.png") ## x in BBDM

        x,y,z = self.load_image(img_path_target,transform),self.load_image(img_path_first,transform),self.load_image(img_path_second,transform)

        return x,y,z


class MidB(Dataset):
    ## interpolation dataset for UCF
    def __init__(self, image_size=(256, 256), flip=False, to_normal=True):
        self.image_size = image_size
        self.cond_dirs = []
        self.gt_dirs = []
        classes = os.listdir('data/MidB/other-data')
        
        for c in classes:
            self.cond_dirs.append(os.path.join('data/MidB/other-data',c))
            self.gt_dirs.append(os.path.join('data/MidB/other-gt-interp',c))

        self._length = len(classes) ## folder of the images
        self.to_normal = to_normal # if normalize to [-1, 1] nor not
        self.flip = flip ## if flip or not

    def __len__(self):
        return self._length

    def load_image(self,img_path,transform):
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)
        
        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        return image

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        img_path_first = os.path.join(self.cond_dirs[index],"frame10.png") ## y in BBDM
        img_path_second = os.path.join(self.cond_dirs[index],"frame11.png") ## z in BBDM
        img_path_target = os.path.join(self.gt_dirs[index],"frame10i11.png") ## x in BBDM

        x,y,z = self.load_image(img_path_target,transform),self.load_image(img_path_first,transform),self.load_image(img_path_second,transform)

        return x,y,z


class DAVIS(Dataset):
    ## interpolation dataset for UCF
    def __init__(self, image_size=(256, 256), flip=False, to_normal=True):
        self.image_size = image_size
        self.image_dirs = []
        for root,dirs,files in os.walk('data/DAVIS'):
            for file in files:
                if 'png' in file or 'jpg' in file:
                    self.image_dirs.append(root)
                    break

        self._length = len(self.image_dirs) ## folder of the images
        self.to_normal = to_normal# if normalize to [-1, 1] nor not
        self.flip = flip ## if flip or not

    def __len__(self):

        return self._length

    def load_image(self,img_path,transform):
        try:
            image = Image.open(img_path)
        except:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)
        
        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)


        return image

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        img_path_first = os.path.join(self.image_dirs[index],"frame_0.jpg") ## y in BBDM
        img_path_second = os.path.join(self.image_dirs[index],"frame_2.jpg") ## z in BBDM
        img_path_target = os.path.join(self.image_dirs[index],"frame_1.jpg") ## x in BBDM

        x,y,z = self.load_image(img_path_target,transform),self.load_image(img_path_first,transform),self.load_image(img_path_second,transform)

        return x,y,z


class FILM(Dataset):
    ## interpolation dataset for UCF
    def __init__(self, image_size=(256, 256), flip=False, to_normal=True,mode = 'easy'):
        self.image_size = image_size
        root = 'data/SNU-FILM'
        file = os.path.join(root,f'test-{mode}.txt')
        f = open(file, "r")
        im_list = f.read()
        self.image_dirs = im_list.split('\n')[:-1]

        self._length = len(self.image_dirs) ## folder of the images
        self.to_normal = to_normal # if normalize to [-1, 1] nor not
        self.flip = flip ## if flip or not

    def __len__(self):

        return self._length

    def load_image(self,img_path,transform):

        try:
            image = Image.open(img_path)
        except:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)
        

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)


        return image

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_path_first,img_path_target,img_path_second = self.image_dirs[index].split(' ') ## y,x,z

        x,y,z = self.load_image(img_path_target,transform),self.load_image(img_path_first,transform),self.load_image(img_path_second,transform)

        return x,y,z



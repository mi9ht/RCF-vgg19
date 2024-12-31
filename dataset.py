from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2



def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class Dataset(data.Dataset):
    def __init__(self, root='data', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        # print(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(cv2.imread(lb_file), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < 102)] = 2
            lb[lb >= 102] = 1

        else:
            img_file = self.filelist[index].split()[0]

        if self.split == "train":
            img = np.array(cv2.imread(img_file), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb
        else:
            img = np.array(Image.open(img_file), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img


if __name__ == '__main__':
    train = Dataset()
    test = Dataset(split="test")
    for img, lb in train:
        print(img.shape, lb.shape)

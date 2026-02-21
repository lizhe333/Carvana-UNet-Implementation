from turtle import Turtle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Unet
from dataset import CarvanaDataset
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim

#定义相关的超参数
LEARNING_RATE = 1e-4
SPLIT=0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 4
NUM_WORKERS = 4
IMAGE_HEIGHT = 572
IMAGE_WIDTH = 572
PIN_MEMORY = True
DATAPATH = "../input/carvana-image-masking-challenge/"
TRAIN_IMG_DIR = './train'
TRAIN_MASK_DIR = './train_masks'

import os
images=os.listdir(TRAIN_IMG_DIR)
masks=os.listdir(TRAIN_MASK_DIR)

import albumentations as A
from albumentations.pytorch import ToTensorV2 
#Compose是将一系列变换操作组合成一个流水线先对图像进行变换，再对掩码进行变换
#有50%的概率进行水平翻转，10%的概率进行垂直翻转
#Normalize是将图像的像素值归一化到[0,1]之间
#ToTensorV2是将图像转换为PyTorch的张量格式
train_transform=A.Compose([
    A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),
    A.Rotate(limit=35,p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])
val_transform=A.Compose([
    A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])
#定义训练集和验证集的划分函数
#splitSize是验证集的比例
def train_test_split(images,splitSize):
    imageLen=len(images)
    val_len=int(imageLen*splitSize)
    train_len=imageLen-val_len
    train_images=images[:train_len]
    val_images=images[train_len:]
    return train_images,val_images

train_images,val_images=train_test_split(images,SPLIT)


train_dataset=CarvanaDataset(train_images,TRAIN_IMG_DIR,TRAIN_MASK_DIR,train_transform,transform=True)
val_dataset=CarvanaDataset(val_images,TRAIN_IMG_DIR,TRAIN_MASK_DIR,val_transform,transform=True)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
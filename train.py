from turtle import Turtle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


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
EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 572
IMAGE_WIDTH = 572
PIN_MEMORY = True
DATAPATH = "../input/carvana-image-masking-challenge/"
TRAIN_IMG_DIR = './dataset/train'
TRAIN_MASK_DIR = './dataset/train_masks'

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
    #先取少量的进行验证
    train_images=train_images[:100]

    val_images=images[train_len:]

    val_images=val_images[:100]
    return train_images,val_images

train_images,val_images=train_test_split(images,SPLIT)


train_dataset=CarvanaDataset(train_images,TRAIN_IMG_DIR,TRAIN_MASK_DIR,train_transform,train=True)
val_dataset=CarvanaDataset(val_images,TRAIN_IMG_DIR,TRAIN_MASK_DIR,val_transform,train=False)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

#定义模型，损失函数，优化器
model=Unet().to(DEVICE)
train_loss=[]
val_loss=[] 
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
criterion=nn.BCEWithLogitsLoss()

def train_one_epoch(model,dataloader,optimizer,criterion,device):
    model.train()
    running_loss=0.0
    counter=0
    for batch in tqdm(dataloader,desc="Training"):
        images=batch["image"].to(device)
        masks=batch["mask"].to(device)

        #先清空梯度
        optimizer.zero_grad()
        #前向传播
        pre=model(images) #[B,1,H,W]
        #计算损失
        #mask:[B,H,W]
        #需要先对pre进行一个squeeze
        pre=pre.squeeze(1) #[B,1,H,W]->[B,H,W]
        loss=criterion(pre,masks)
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
        running_loss+=loss.item()
        counter+=1
        #计算每个epoch的平均损失
    epoch_loss=running_loss/counter
    return epoch_loss

def eval_one_epoch(model,dataloader,criterion,device):
    model.eval()
    running_loss=0.0
    counter=0
    with torch.no_grad():
        #不记录梯度，只执行前向传播
        for batch in tqdm(dataloader,desc="Evaluating"):
            images=batch["image"].to(device)
            masks=batch["mask"].to(device)

            #前向传播
            pre=model(images) #[B,1,H,W]
            #计算损失
            #mask:[B,H,W]
            #需要先对pre进行一个squeeze
            pre=pre.squeeze(1) #[B,1,H,W]->[B,H,W]
            
            loss=criterion(pre,masks)
            running_loss+=loss.item()
            counter+=1
    epoch_loss=running_loss/counter
    return epoch_loss

train_loss=[]
val_loss=[]
if __name__ == "__main__":
    model=Unet().to(DEVICE)
    for epoch in range(EPOCHS):
        train_loss=train_one_epoch(model,train_loader,optimizer,criterion,DEVICE)
        val_loss=eval_one_epoch(model,val_loader,criterion,DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
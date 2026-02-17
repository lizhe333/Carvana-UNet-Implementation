import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# Carvana Dataset 处理数据集，读取图像掩码，增强图像，转换为张量
class CarvanaDataset(Dataset):
    def __init__(self,images,img_dir,mask_dir,transform=None,train=True):
        """
        Args:
            images : 图像文件名列表
            img_dir : 图像目录
            mask_dir (str): 掩码目录
            transform: 图像增强操作
        """
        self.images = images
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        """
        Args:
            index: 文件名称
        Returns:
            dict: {"image": ..., "mask": ...}
        """
        img_path=os.path.join(self.img_dir,self.images[index])
        mask_path=os.path.join(self.mask_dir,self.images[index].replace(".jpg","_mask.gif"))

        #读取图片和掩码
        image=np.array(Image.open(img_path).convert("RGB"))
        mask=np.array(Image.open(mask_path).convert("L"),dtype=np.float32)  
        #将掩码的无效值转化成1.0
        mask[mask==255]=1.0

        #应用图像增强
        if  self.transform:
            augmented=self.transform(image=image,mask=mask)
            #从增强后的字典中提取图像和掩码来覆盖掉原来的值
            image=augmented["image"]
            mask=augmented["mask"]
        return {"image":image,"mask":mask}


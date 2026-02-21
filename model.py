#此处使用的是更加现代化的U-Net，增加了batchnorm同时每个蓝色箭头之间不改变图像的尺寸
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def double_conv(in_channels,out_channels):
    '''
    双卷积层
    '''
    conv=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,1,1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,1,1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv.to(DEVICE)

#当遇到尺寸不一致的时候，用一个强制补0的方法来解决,创建一个和目标一样大的全0,这样小的特征图直接复制到新的tensor上，
#不全的部分就会被填补成0
def addPadding(scrShapeTensor,tensor_whose_shape_is_to_be_changed):
    if scrShapeTensor!=tensor_whose_shape_is_to_be_changed.shape:
        target=torch.zeros(scrShapeTensor.shape)#B,C,H,W
        target[:,:,
                :tensor_whose_shape_is_to_be_changed.shape[2],
                :tensor_whose_shape_is_to_be_changed.shape[3]]=tensor_whose_shape_is_to_be_changed
        return target
    else:
        return tensor_whose_shape_is_to_be_changed

#拼接Unet
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.max_pool_2x2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.dconv_down1=double_conv(3,64)
        self.dconv_down2=double_conv(64,128)
        self.dconv_down3=double_conv(128,256)
        self.dconv_down4=double_conv(256,512)
        self.dconv_down5=double_conv(512,1024)

        #使用转置卷积来进行上采样
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_conv_1=double_conv(1024,512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

    def forward(self,image):
        #编码器
        x1=self.dconv_down1(image)
        #print("x1:"+str(x1.shape))
        x2=self.max_pool_2x2(x1)
        #print("x2:"+str(x2.shape))
        x3=self.dconv_down2(x2)
        #print("x3:"+str(x3.shape))
        x4=self.max_pool_2x2(x3)
        #print("x4:"+str(x4.shape))
        x5=self.dconv_down3(x4)
        #print("x5:"+str(x5.shape))
        x6=self.max_pool_2x2(x5)
        #print("x6:"+str(x6.shape))
        x7=self.dconv_down4(x6)
        #print("x7:"+str(x7.shape))
        x8=self.max_pool_2x2(x7)
        #print("x8:"+str(x8.shape))
        x9=self.dconv_down5(x8)
        #print("x9:"+str(x9.shape))

        #解码器
        x=self.up_trans_1(x9)
        x=addPadding(x7,x)
        x=torch.cat([x7,x],dim=1)
        #print("x:"+str(x.shape))
        x=self.up_conv_1(x)
    
        x=self.up_trans_2(x)
        x=addPadding(x5,x)
        x=torch.cat([x5,x],dim=1)
        #print("x:"+str(x.shape))
        x=self.up_conv_2(x)

        x=self.up_trans_3(x)
        x=addPadding(x3,x)
        x=torch.cat([x3,x],dim=1)
        #print("x:"+str(x.shape))
        x=self.up_conv_3(x)
        
        x=self.up_trans_4(x)
        x=addPadding(x1,x)
        x=torch.cat([x1,x],dim=1)   
        x=self.up_conv_4(x)
        #print("x:"+str(x.shape))
        
        x=self.out(x)
        
        return x.to(DEVICE)

# if __name__ == '__main__':
#     image=torch.randn(1,3,572,572)
#     model = Unet()
#     model(image)
# ImageSegmentation With Vnet3D
> This is an example of the CT images Segment from LiTS---Liver-Tumor-Segmentation-Challenge
![](LiTS_header.jpg)

## How to Use
(re)implemented the model with tensorflow in the paper of "Milletari, F., Navab, N., & Ahmadi, S. A. (2016) V-net: Fully convolutional neural networks for volumetric medical image segmentation.3DV 2016"

**1、Preprocess**
* LiTS data of image and mask are all type of .nii files,in order to train and visulise,convert .nii file to .bmp file. 
* Liver data preparing,i have tried many patch size,and finally using the patch(256,256,16),if you have better GPU,you can change 16 to 24 or 32:run the getPatchImageAndMask.py
* Tumor data preparing,using the patch(256,256,16):run the getPatchImageAndMask.py,disable the line gen_image_mask(srcimg, seg_liverimage, i, shape=(16, 256, 256), numberxy=5, numberz=10) and enable the line gen_image_mask(srcimg, seg_tumorimage, i, shape=(16, 256, 256), numberxy=5, numberz=10),and change the trainLiverMask to trainTumorMask
* last save all the data folder path into csv file: run the utils.py

the file like this:
G:\Data\segmentation\Image/0_161
G:\Data\segmentation\Image/0_162
G:\Data\segmentation\Image/0_163


**2、Liver and Tumor Segmentation**
* the VNet model

![](3dVNet.png) 

* train and predict in the script of vnet3d_train.py and vnet3d_predict.py

**3、download resource**
* liver segmentation trained model,log,test data can download on here:https://pan.baidu.com/s/1ijK6BG3vZM4nHwZ6S2yFiw, password：74j5 
* LiTS data have 130 cases,using 0-110 cases trainging,and other is testing.testing result can download on here:https://pan.baidu.com/s/1A_-u7tJcn7rIqnrLaSqi4A password：22es 

## Result
Trained Loss
![](diceloss.PNG)

Liver Segment Result

Liver leaderboard
![](livertop30.PNG)

test case segmentation result can see in the file of 35.mp4,38.mp4 and 51.mp4

first col is srcimage,second col is GroundTruth Mask image,third col is VNet segmentation image
![](GTvsVNet.bmp)

Lesion leaderboard
![](tumortop34.PNG)

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com,ydx0902@gmail.com,188123134@qq.com
* Contact:junqiangChen,dexianYe,xingTao
* WeChat Public number: 最新医学影像技术

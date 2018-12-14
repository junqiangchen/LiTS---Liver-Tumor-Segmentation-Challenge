import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
import numpy as np
import pandas as pd
import cv2


def predict():
    height = 512
    width = 512
    dimension = 32
    Vnet3d = Vnet3dModule(height, width, dimension, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\\diceVnet3d\\model\Vnet3d.pd")
    srcimagepath = "D:\Data\LIST\\test\Image\\111"
    predictpath = "D:\Data\LIST\\test\PredictMask"
    index = 0
    imagelist = []
    for _ in os.listdir(srcimagepath):
        image = cv2.imread(srcimagepath + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
        tmpimage = np.reshape(image, (height, width, 1))
        imagelist.append(tmpimage)
        index += 1

    imagearray = np.array(imagelist)
    imagearray = np.reshape(imagearray, (index, height, width, 1))
    imagemask = np.zeros((index, height, width), np.int32)

    for i in range(0, index + dimension, dimension // 2):
        if (i + dimension) <= index:
            imagedata = imagearray[i:i + dimension, :, :, :]
            imagemask[i:i + dimension, :, :] = Vnet3d.prediction(imagedata)
        elif (i < index):
            imagedata = imagearray[index - dimension:index, :, :, :]
            imagemask[index - dimension:index, :, :] = Vnet3d.prediction(imagedata)

    mask = imagemask.copy()
    mask[imagemask > 0] = 255
    result = np.clip(mask, 0, 255).astype('uint8')
    for i in range(0, index):
        cv2.imwrite(predictpath + "/" + str(i) + ".bmp", result[i])


predict()

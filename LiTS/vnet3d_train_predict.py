from promise2012.Vnet.model_vnet3d import Vnet3dModule
from promise2012.Vnet.util import convertMetaModelToPbModel
import numpy as np
import pandas as pd
import cv2


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('trainY.csv')
    csvimagedata = pd.read_csv('trainX.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    Vnet3d = Vnet3dModule(128, 128, 64, channels=1, costname="dice coefficient")
    Vnet3d.train(imagedata, maskdata, "model\\Vnet3dModule.pd", "log\\", 0.001, 0.7, 100000, 1)


def predict0():
    Vnet3d = Vnet3dModule(256, 256, 64, inference=True, model_path="model\\Vnet3dModule.pd")
    for filenumber in range(30):
        batch_xs = np.zeros(shape=(64, 256, 256))
        for index in range(64):
            imgs = cv2.imread(
                "D:\Data\PROMISE2012\Vnet3d_data\\test\image\\" + str(filenumber) + "\\" + str(index) + ".bmp", 0)
            batch_xs[index, :, :] = imgs[128:384, 128:384]

        predictvalue = Vnet3d.prediction(batch_xs)

        for index in range(64):
            result = np.zeros(shape=(512, 512), dtype=np.uint8)
            result[128:384, 128:384] = predictvalue[index]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(
                "D:\Data\PROMISE2012\Vnet3d_data\\test\image\\" + str(filenumber) + "\\" + str(index) + "mask.bmp",
                result)


def meta2pd():
    convertMetaModelToPbModel(meta_model="model\\Vnet3dModule.pd", pb_model="model")
            
train()
#predict0()
#meta2pd()

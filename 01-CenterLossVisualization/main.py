import sys, os
import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import Model
from Base import Utility, Train, Model
# https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization


strPtojectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strPtojectFolderPath, "01-RAWData")

# Hyper Parameter
dictHyperPara = {}
dictHyperPara["intBatchSize"] = 128
dictHyperPara["intEpochs"] = 30

# Model Parameter
dictModelPara = {}
dictModelPara["intImageW"] = 28 
dictModelPara["intImageH"] = 28 
dictModelPara["intImageC"] = 1 
dictModelPara["intConvL1"] = 32 
dictModelPara["intConvL2"] = 64
dictModelPara["intKernelSize1"] = (5, 5)
dictModelPara["intKernelSize2"] = (3, 3)
dictModelPara["intHiddenSize"] = 2
dictModelPara["intClassesNum"] = 10
dictModelPara["boolCenterLoss"] = True


(arrayTrainX, arrayTrainY), (arrayValidX, arrayValidY) = mnist.load_data()

if K.image_data_format() == "channels_first":
    arrayTrainX = arrayTrainX.reshape(arrayTrainX.shape[0], dictModelPara["intImageC"], dictModelPara["intImageW"], dictModelPara["intImageH"])
    arrayValidX = arrayValidX.reshape(arrayValidX.shape[0], dictModelPara["intImageC"], dictModelPara["intImageW"], dictModelPara["intImageH"])
else:
    arrayTrainX = arrayTrainX.reshape(arrayTrainX.shape[0], dictModelPara["intImageW"], dictModelPara["intImageH"], dictModelPara["intImageC"])
    arrayValidX = arrayValidX.reshape(arrayValidX.shape[0], dictModelPara["intImageW"], dictModelPara["intImageH"], dictModelPara["intImageC"])


arrayTrainX = arrayTrainX.astype("float32")/255.
arrayValidX = arrayValidX.astype("float32")/255.

arrayTrainOhY = keras.utils.to_categorical(arrayTrainY, dictModelPara["intClassesNum"])
arrayValidOhY = keras.utils.to_categorical(arrayValidY, dictModelPara["intClassesNum"])
print("x_train shape:", arrayTrainX.shape)
print(arrayTrainX.shape[0], "train samples")
print(arrayValidX.shape[0], "test samples")

Train.getTrain(dictModelPara=dictModelPara, dictHyperPara=dictHyperPara, arrayTrain=(arrayTrainX, arrayTrainOhY, arrayTrainY), arrayValid=(arrayValidX, arrayValidOhY, arrayValidY))


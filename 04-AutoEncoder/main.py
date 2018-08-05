
import os
import tensorflow as tf
from Base import DataProcessing, Model
import matplotlib.pyplot as plt
import numpy as np


strProjectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

# prepare data
ETL = DataProcessing.prepareData()
ETL.cleanData(strDataFileName="train.csv", boolLabel=True)
ETL.doNormalize()
mnist = ETL.dictData["Data"]

# hyperparameter
n_input = 784
folat_Learning_rate = 0.01
int_Epochs = 100
int_Batch_size = 200

# model
AutoEncoder = Model.AutoEncoder(n_input=n_input
                              , float_Learning_rate=folat_Learning_rate
                              , n_hidden=[256, 128, 2])

# training
list_Loss = AutoEncoder.fit(X=mnist
                          , int_Epochs=int_Epochs
                          , int_Batch_size=int_Batch_size)

plt.plot(np.arange(len(list_Loss)), list_Loss)
plt.savefig(os.path.join(strOutputFolderPath, "LossCurve_lr={}.jpg").format(folat_Learning_rate))

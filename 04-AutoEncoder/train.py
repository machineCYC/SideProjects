import os
from Base import DataProcessing, Model
import matplotlib.pyplot as plt
import numpy as np


strProjectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

# prepare data
ETL = DataProcessing.prepareData()

# preapre train data
ETL.cleanData(strDataFileName="train.csv", boolLabel=True, boolNormal=True)
mnist_train = ETL.dictData["Data"]

# preapre vaild data
ETL.cleanData(strDataFileName="test.csv", boolLabel=False, boolNormal=True)
mnist_test = ETL.dictData["Data"]

# hyperparameter
n_input = 784
n_hidden = [256, 64, 2]
folat_Learning_rate = 0.0005
int_Epochs = 150
int_Batch_size = 200

# model
AutoEncoder = Model.AutoEncoder(n_input=n_input
                              , n_hidden=n_hidden
                              , float_Learning_rate=folat_Learning_rate)

# training
AutoEncoder.fit(X=mnist_train
              , Y=mnist_train
              , int_Epochs=int_Epochs
              , int_Batch_size=int_Batch_size
              , validation_data=(mnist_test, mnist_test))

print("Run the command line:\n" \
      "--> tensorboard --logdir=\"Tensorboard/train/\", Tensorboard/vaild/\"" \
      "\nThen open http://asus:6006 into your web browser")


# save model
AutoEncoder.save(path=os.path.join(strOutputFolderPath, "AutoEncoder"))
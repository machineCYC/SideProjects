
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
ETL.cleanData(strDataFileName="train.csv", boolLabel=True, boolNormal=True)
mnist = ETL.dictData["Data"]

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
list_Loss = AutoEncoder.fit(X=mnist
                          , Y=mnist
                          , int_Epochs=int_Epochs
                          , int_Batch_size=int_Batch_size)

plt.plot(np.arange(len(list_Loss)), list_Loss)
plt.savefig(os.path.join(strOutputFolderPath, "LossCurve_lr={}.jpg").format(folat_Learning_rate))

AutoEncoder.save(path=os.path.join(strOutputFolderPath, "AutoEncoder"))

AutoEncoder.reload(path=os.path.join(strOutputFolderPath, "AutoEncoder"))

# reconstruct images
fig, axis = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    img_train_org = mnist[i].reshape(28, 28)
    axis[0][i].imshow(img_train_org, cmap="gray")

    img_train = np.reshape(AutoEncoder.predict(mnist[i].reshape(-1, 784)), (28, 28))
    axis[1][i].imshow(img_train, cmap="gray")
plt.savefig(os.path.join(strOutputFolderPath, "recons_Train.jpg"))


import os
from Base import DataProcessing, Model
import matplotlib.pyplot as plt
import numpy as np


strProjectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

# prepare data
ETL = DataProcessing.prepareData()
ETL.cleanData(strDataFileName="test.csv", boolLabel=False, boolNormal=True)
mnist_test = ETL.dictData["Data"]

# hyperparameter
n_input = 784
n_hidden = [256, 64, 2]
folat_Learning_rate = 0.0005

# model
AutoEncoder = Model.AutoEncoder(n_input=n_input
                              , n_hidden=n_hidden
                              , float_Learning_rate=folat_Learning_rate)

# reload model
AutoEncoder.reload(path=os.path.join(strOutputFolderPath, "AutoEncoder"))

# reconstruct testing images
fig, axis = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    img_test_org = mnist_test[i].reshape(28, 28)
    axis[0][i].imshow(img_test_org, cmap="gray")

    img_test = np.reshape(AutoEncoder.predict(mnist_test[i].reshape(-1, 784)), (28, 28))
    axis[1][i].imshow(img_test, cmap="gray")
plt.savefig(os.path.join(strOutputFolderPath, "recons_Test.jpg"))
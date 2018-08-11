import os
import numpy as np

from Base import DataProcessing, Model, Utility


strProjectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

# prepare data
ETL = DataProcessing.prepareData()

# prepare train data
ETL.cleanData(strDataFileName="train.csv", boolLabel=True, boolNormal=True)
mnist_train = ETL.dictData["Data"]
mnist_train_label = ETL.dictData["Label"]

# prepare test data
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

# reconstruct training images
Utility.visualize_reconstrict_images(AutoEncoder, "recons_Train.jpg", mnist_train)
# reconstruct testing images
Utility.visualize_reconstrict_images(AutoEncoder, "recons_Test.jpg", mnist_test)

# visualize code
Utility.visualize_2dim_code(AutoEncoder, "2dim_train.jpg", mnist_train, mnist_train_label)

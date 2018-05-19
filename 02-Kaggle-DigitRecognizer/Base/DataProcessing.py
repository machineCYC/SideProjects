import os
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")


class prepareData():
    def __init__(self):
        self.dictData = {} 

    def cleanData(self, strDataFileName, boolLabel):
        pdData = pd.read_csv(os.path.join(strRAWDataFolderPath, strDataFileName))

        if boolLabel:
            self.dictData["Data"] = [pdData.drop(labels=["label"], axis=1).values, pdData["label"].values]
        else:
            self.dictData["Data"] = [pdData.values]

    def doNormalize(self):
        for key in self.dictData:
            self.dictData[key][0] = self.dictData[key][0]/255.0

    def convertVector2Image(self, intImageW, intImageH, intImageC):
        for key in self.dictData:
            self.dictData[key][0] = self.dictData[key][0].reshape(-1, intImageW, intImageH, intImageC)

    def convertLabel2Onehot(self):
        for key in self.dictData:
            if len(self.dictData[key]) == 2:
                self.dictData[key][1] = np.array(to_categorical(self.dictData[key][1]))

    def splitData(self, floatRatio):
        data = self.dictData["Data"]
        X = data[0]
        Y = data[1]
        intDataSize = len(X)
        intValidationSize = int(intDataSize * floatRatio)
        return (X[intValidationSize:], Y[intValidationSize:]), (X[:intValidationSize], Y[:intValidationSize])

    def getData(self):
        return self.dictData["Data"]


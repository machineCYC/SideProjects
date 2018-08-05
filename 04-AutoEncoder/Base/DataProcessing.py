import os
import numpy as np
import pandas as pd


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")


class prepareData(object):
    """
    Excute all process before training model
    """

    def __init__(self):
        """
        Initialize a cleaning data
        """
        self.dictData = {} 

    def cleanData(self, strDataFileName, boolLabel):
        """
        Split the data and label

        Inputs:
        - strDataFileName: A str of data file name.
        - boolLabel: If true, will create a key names Label to storage the label.
        """
        pdData = pd.read_csv(os.path.join(strRAWDataFolderPath, strDataFileName))

        if boolLabel:
            self.dictData["Data"] = pdData.drop(labels=["label"], axis=1).values
            self.dictData["Label"] = pdData["label"].values
        else:
            self.dictData["Data"] = pdData.values

    def doNormalize(self):
        """
        This funciton divide 255 for all pixel
        """

        self.dictData["Data"] = self.dictData["Data"] / 255.0


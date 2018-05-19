import os
import numpy as np
import pandas as pd
from Base import DataProcessing, Train, Predict


def getTest():

    strProjectFolder = os.path.dirname(__file__)
    strRAWDataFolder = os.path.join(strProjectFolder, "01-RAWData")
    strOutputFolder = os.path.join(strProjectFolder, "02-Output")
    submisson = pd.read_csv(os.path.join(strRAWDataFolder, "sample_submission.csv"))

    ETL = DataProcessing.prepareData()
    ETL.cleanData(strDataFileName="Train.csv", boolLabel=True)

   # Hyper Parameter
    dictHyperPara = {}
    dictHyperPara["intBatchSize"] = 128
    dictHyperPara["intEpochs"] = 30
    dictHyperPara["boolTrain"] = False

    # Model Parameter
    dictModelPara = {}
    dictModelPara["intImageW"] = 28
    dictModelPara["intImageH"] = 28
    dictModelPara["intImageC"] = 1

    dictModelPara["intConvL1"] = 32
    dictModelPara["intPoolSize1"] = (2, 2)
    dictModelPara["intStrides1"] = 2
    dictModelPara["intKernelSize1"] = (5, 5)
    dictModelPara["strPaddingType1"] = "same"

    dictModelPara["intConvL2"] = 64
    dictModelPara["intPoolSize2"] = (2, 2)
    dictModelPara["intStrides2"] = 2
    dictModelPara["intKernelSize2"] = (3, 3)
    dictModelPara["strPaddingType2"] = "same"

    dictModelPara["intHiddenSize1"] = 64
    dictModelPara["intHiddenSize2"] = 64
    dictModelPara["intClassesNum"] = 10


    ETL.convertVector2Image(intImageW=dictModelPara["intImageW"], intImageH=dictModelPara["intImageH"], intImageC=dictModelPara["intImageC"])
    ETL.doNormalize()
    ETL.convertLabel2Onehot()

    arrayTrain = ETL.getData()
    Train.getTrain(dictModelPara, dictHyperPara, arrayTrain, arrayTrain)

    ETL.cleanData(strDataFileName="Test.csv", boolLabel=False)
    ETL.convertVector2Image(intImageW=dictModelPara["intImageW"], intImageH=dictModelPara["intImageH"], intImageC=dictModelPara["intImageC"])
    arrayTest = ETL.getData()

    arrayPredict = Predict.makePredict(arrayTest, strOutputFolder)

    submisson["Label"] = pd.DataFrame(arrayPredict)
    submisson.to_csv(os.path.join(strOutputFolder, "submission.csv"), index=False)

if __name__ == "__main__":
    getTest()
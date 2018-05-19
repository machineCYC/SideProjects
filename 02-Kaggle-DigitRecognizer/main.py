import sys, os
from Base import DataProcessing, Train


strPtojectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strPtojectFolderPath, "01-RAWData")


ETL = DataProcessing.prepareData()
ETL.cleanData(strDataFileName="train.csv", boolLabel=True)

floatRatio = 0.1

# Hyper Parameter
dictHyperPara = {}
dictHyperPara["intBatchSize"] = 128
dictHyperPara["intEpochs"] = 30
dictHyperPara["boolTrain"] = True

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
ETL.convertLabel2Onehot()
ETL.doNormalize()
arrayTrain, arrayValid = ETL.splitData(floatRatio=floatRatio)

Train.getTrain(dictModelPara, dictHyperPara, arrayTrain, arrayValid)
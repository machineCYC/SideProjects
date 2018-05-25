import sys, os
from Base import DataProcessing, Train, Utility
import numpy as np

strProjectFolderPath = os.path.dirname(__file__)
strRAWDataFolderPath = os.path.join(strProjectFolderPath, "01-RAWData")
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

ETL = DataProcessing.prepareData()

# prepare data
ETL.cleanData(strDataFileName="train.csv", boolLabel=True)

floatRatio = 0.1

# Hyper Parameter
dictHyperPara = {}
dictHyperPara["intBatchSize"] = 256
dictHyperPara["intEpochs"] = 30
dictHyperPara["boolTrain"] = True

# Model Parameter
dictModelPara = {}
dictModelPara["intImageW"] = 28
dictModelPara["intImageH"] = 28
dictModelPara["intImageC"] = 1

dictModelPara["intConvL1"] = 16
dictModelPara["intPoolSize1"] = (2, 2)
dictModelPara["intStrides1"] = 2
dictModelPara["intKernelSize1"] = (3, 3)
dictModelPara["strPaddingType1"] = "same"

dictModelPara["intConvL2"] = 32
dictModelPara["intPoolSize2"] = (2, 2)
dictModelPara["intStrides2"] = 2
dictModelPara["intKernelSize2"] = (3, 3)
dictModelPara["strPaddingType2"] = "same"

dictModelPara["intHiddenSize1"] = 64
dictModelPara["intClassesNum"] = 10

# data processing
ETL.convertVector2Image(intImageW=dictModelPara["intImageW"], intImageH=dictModelPara["intImageH"], intImageC=dictModelPara["intImageC"])
ETL.convertLabel2Onehot()
ETL.doNormalize()
arrayTrain, arrayValid = ETL.splitData(floatRatio=floatRatio)

# Visualizing Filters
VisualizingFilters = Utility.VisualizingFilters(boolTrain=dictHyperPara["boolTrain"], dictModelPara=dictModelPara, strOutputFolderPath=strOutputFolderPath)
VisualizingFilters.plotSaliencyMap(arrayX=arrayValid[0], arrayYLabel=np.argmax(arrayValid[1], axis=-1))
VisualizingFilters.plotImageFiltersResult(arrayValid[0], intChooseId=0)
VisualizingFilters.plotWhiteNoiseActivateFilters()

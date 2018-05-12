import os
import numpy as np
from Base import Model, Utility
from keras.callbacks import ModelCheckpoint, CSVLogger


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strOutputFolderPath = os.path.join(strProjectFolderPath, "Output")

def getTrain(dictModelPara, dictHyperPara, arrayTrain, arrayValid):

    history = Utility.Records(boolCenterLoss=dictModelPara["boolCenterLoss"], strOutputFolderPath=strOutputFolderPath)

    model = Model.CNN(dictModelPara)

    if dictModelPara["boolCenterLoss"]:
        strSavePath = os.path.join(os.path.join(strOutputFolderPath, "IsCenter"))
    else:
        strSavePath = os.path.join(os.path.join(strOutputFolderPath, "NonCenter"))

    callbacks = [ModelCheckpoint(os.path.join(strSavePath, "model.h5"), save_best_only=True)
                , CSVLogger(os.path.join(strSavePath, "log.csv"), separator=",", append=False)
                , history]

    if dictModelPara["boolCenterLoss"]:
        arrayRandomTrainY = np.random.rand(arrayTrain[0].shape[0], 1)
        arrayRandomValidY = np.random.rand(arrayValid[0].shape[0], 1)

        model.fit(x=[arrayTrain[0], arrayTrain[2]], y=[arrayTrain[1], arrayRandomTrainY], epochs=dictHyperPara["intEpochs"], batch_size=dictHyperPara["intBatchSize"], verbose=2, validation_data=([arrayValid[0], arrayValid[2]], [arrayValid[1], arrayRandomValidY]), callbacks=callbacks)
    else:
        model.fit(x=arrayTrain[0], y=arrayTrain[1], epochs=dictHyperPara["intEpochs"], batch_size=dictHyperPara["intBatchSize"], verbose=2, validation_data=arrayValid, callbacks=callbacks)

    history.plotLosss()
    model.save(os.path.join(strSavePath, "model.h5"))
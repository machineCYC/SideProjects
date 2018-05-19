import os
from Base import Model, Utility
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

def getTrain(dictModelPara, dictHyperPara, arrayTrain, arrayValid):
    history = Utility.recordsAccLossHistory(boolTrain=dictHyperPara["boolTrain"], strOutputFolderPath=strOutputFolderPath)

    if dictHyperPara["boolTrain"]:
        strSaveFolderPath = os.path.join(strOutputFolderPath, "01Train")
    else:
        strSaveFolderPath = os.path.join(strOutputFolderPath, "02Test")

    model = Model.CNN(dictModelPara)

    callbacks = [EarlyStopping("val_loss", patience=20)
               , ModelCheckpoint(os.path.join(strSaveFolderPath, "model.h5"), save_best_only=True)
               , CSVLogger(os.path.join(strSaveFolderPath, "log.csv"), separator=",", append=False)
               , history]

    model.fit(x=arrayTrain[0], y=arrayTrain[1], epochs=dictHyperPara["intEpochs"], batch_size=dictHyperPara["intBatchSize"], verbose=2, validation_data=arrayValid, callbacks=callbacks)

    history.plotLosss()
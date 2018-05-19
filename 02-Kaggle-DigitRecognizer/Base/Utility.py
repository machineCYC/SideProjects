import os, keras
import numpy as np
import matplotlib.pyplot as plt


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

class recordsAccLossHistory(keras.callbacks.Callback):
    def __init__(self, boolTrain, strOutputFolderPath):
        self.boolTrain = boolTrain
        if self.boolTrain:
            self.strSaveFolderPath = os.path.join(strOutputFolderPath, "01Train")
        else:
            self.strSaveFolderPath = os.path.join(strOutputFolderPath, "02Test")

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_losses = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs["loss"])
        self.accuracy.append(logs["acc"])
        self.val_losses.append(logs["val_loss"])
        self.val_accuracy.append(logs["val_acc"])

    def on_train_end(self, logs={}):
        pass

    def plotLosss(self):
        fig = plt.figure(figsize=(12, 5))
        # Loss Curves
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(np.arange(len(self.losses)), self.losses, label="losses")
        plt.plot(np.arange(len(self.val_losses)), self.val_losses, label="val_losses")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.title("loss process")
        plt.tight_layout()
        # Accuracy Curves
        ax = fig.add_subplot(1, 2, 2)
        plt.plot(np.arange(len(self.accuracy)), self.accuracy, label="accuracy")
        plt.plot(np.arange(len(self.val_accuracy)), self.val_accuracy, label="val_accuracy")
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.legend()
        plt.title("accuracy process")
        plt.tight_layout()
        plt.savefig(os.path.join(self.strSaveFolderPath, "LossAccuracyCurves"))
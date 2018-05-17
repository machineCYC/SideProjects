import numpy as np
import matplotlib.pyplot as plt
import os, keras
from keras.models import Model


def VisualizeCoordinate(arrayOutputs, arrayLabels, epoch, folatLambda, strOutputFolderPath):

    listColor = ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff",
                 "#ff00ff", "#990000", "#999900", "#009900", "#009999"]
    plt.clf()
    for i in range(10):
        plt.plot(arrayOutputs[arrayLabels == i, 0], arrayOutputs[arrayLabels == i, 1], ".", c=listColor[i])
    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
    XMax = np.max(arrayOutputs[:,0]) 
    XMin = np.min(arrayOutputs[:,1])
    YMax = np.max(arrayOutputs[:,0])
    YMin = np.min(arrayOutputs[:,1])

    # plt.xlim(xmin=XMin, xmax=XMax)
    # plt.ylim(ymin=YMin, ymax=YMax)
    plt.text(XMin, YMax, "lambda={} epoch={}".format(folatLambda, epoch))
    plt.savefig(os.path.join(strOutputFolderPath, "lambda={}epoch={}.jpg").format(folatLambda, epoch))


class Records(keras.callbacks.Callback):
	def __init__(self, boolCenterLoss, folatLambda, strOutputFolderPath):
		self.boolCenterLoss = boolCenterLoss
		if self.boolCenterLoss:
			self.strSaveFolderPath = os.path.join(strOutputFolderPath, "IsCenter")
			self.folatLambda = folatLambda
		else:
			self.strSaveFolderPath = os.path.join(strOutputFolderPath, "NonCenter")
			self.folatLambda = 0

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []
		self.val_losses = []
		self.val_accuracy = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		if self.boolCenterLoss:
			self.losses.append(logs["loss"])
			self.accuracy.append(logs["outputs_acc"])
			self.val_losses.append(logs["val_loss"])
			self.val_accuracy.append(logs["val_outputs_acc"])
		else:
			self.losses.append(logs["loss"])
			self.accuracy.append(logs["acc"])
			self.val_losses.append(logs["val_loss"])
			self.val_accuracy.append(logs["val_acc"])
		
		#(IMPORTANT) Only use one input: "inputs=self.model.input[0]"
		inputs = self.model.input # this can be a list or a matrix. 
		if self.boolCenterLoss:
			inputs = self.model.input[0]
			arrayLabels = self.validation_data[1].flatten()
		else:
			arrayLabels = np.argmax(self.validation_data[1], axis=1)
		
		VisualLayerModel = Model(inputs=inputs, outputs=self.model.get_layer("VisualLayer").output)
		arrayOutputs = VisualLayerModel.predict(self.validation_data[0])
		
		VisualizeCoordinate(arrayOutputs, arrayLabels, epoch, self.folatLambda, self.strSaveFolderPath)
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
	
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
		plt.savefig(os.path.join(self.strSaveFolderPath, "LossAccuracyCurvesLambda={}.jpg").format(self.folatLambda))

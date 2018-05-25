import os, keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
import keras.backend as K 


def deprocessImage(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x

def makeNormalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def trainGradAscent(intIterationSteps, arrayInputImageData, targetFunction, intRecordFrequent):
    """
    Implement gradient ascent in targetFunction
    """
    listFilterImages = []
    floatLearningRate = 1e-2
    for i in range(intIterationSteps):
        floatLossValue, arrayGradientsValue = targetFunction([arrayInputImageData, 0])
        arrayInputImageData += arrayGradientsValue * floatLearningRate
        if i % intRecordFrequent == 0:
            listFilterImages.append((arrayInputImageData, floatLossValue))
            print("#{}, loss rate: {}".format(i, floatLossValue))
    return listFilterImages

def plotModel(strSaveFolderPath):
    """
    This function plots the model structure.
    """
    model = load_model(os.path.join(strSaveFolderPath, "model.h5"))
    model.summary()
    plot_model(model, show_shapes=True, to_file=os.path.join(strSaveFolderPath, "model.png"))


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
        plotModel(self.strSaveFolderPath)

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

class VisualizingFilters():
    def __init__(self, boolTrain, dictModelPara, strOutputFolderPath):
        self.boolTrain = boolTrain
        self.dictModelPara = dictModelPara
        if self.boolTrain:
            self.strSaveFolderPath = os.path.join(strOutputFolderPath, "01Train")
        else:
            self.strSaveFolderPath = os.path.join(strOutputFolderPath, "02Test")

    def plotWhiteNoiseActivateFilters(self):
        """
        This function plot Activate Filters with white noise as input images
        """
        intRecordFrequent = 20
        intNumberSteps = 160
        intIterationSteps = 160

        model = load_model(os.path.join(self.strSaveFolderPath, "model.h5"))
        dictLayer = dict([layer.name, layer] for layer in model.layers)
        inputImage = model.input
        listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
        listCollectLayers = [dictLayer[name].output for name in listLayerNames]

        for cnt, fn in enumerate(listCollectLayers):
            listFilterImages = []
            intFilters = 16
            for i in range(intFilters):
                arrayInputImage = np.random.random((1, self.dictModelPara["intImageW"], self.dictModelPara["intImageH"], self.dictModelPara["intImageC"])) # random noise
                tensorTarget = K.mean(fn[:, :, :, i])

                tensorGradients = makeNormalize(K.gradients(tensorTarget, inputImage)[0])
                targetFunction = K.function([inputImage, K.learning_phase()], [tensorTarget, tensorGradients])

                # activate filters
                listFilterImages.append(trainGradAscent(intIterationSteps, arrayInputImage, targetFunction, intRecordFrequent))
            
            for it in range(intNumberSteps//intRecordFrequent):
                print("In the #{}".format(it))
                fig = plt.figure(figsize=(16, 17))
                for i in range(intFilters):
                    ax = fig.add_subplot(intFilters/4, 4, i+1)
                    arrayRawImage = listFilterImages[i][it][0].squeeze()
                    ax.imshow(deprocessImage(arrayRawImage), cmap="Blues")
                    plt.xticks(np.array([]))
                    plt.yticks(np.array([]))
                    plt.xlabel("{:.3f}".format(listFilterImages[i][it][1]))
                    plt.tight_layout()
                fig.suptitle("Filters of layer {} (# Ascent Epoch {} )".format(listLayerNames[cnt], it*intRecordFrequent))
                plt.savefig(os.path.join(self.strSaveFolderPath, "FiltersWhiteNoise" + listLayerNames[cnt]))

    def plotImageFiltersResult(self, arrayX, intChooseId):
        """
        This function plot the output of convolution layer in valid data image.
        """

        model = load_model(os.path.join(self.strSaveFolderPath, "model.h5"))
        dictLayer = dict([layer.name, layer] for layer in model.layers)
        inputImage = model.input
        listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
        # define the function that input is an image and calculate the image through each layer until the output layer that we choose
        listCollectLayers = [K.function([inputImage, K.learning_phase()], [dictLayer[name].output]) for name in listLayerNames] 

        for cnt, fn in enumerate(listCollectLayers):
            arrayPhoto = arrayX[intChooseId].reshape(1, self.dictModelPara["intImageW"], self.dictModelPara["intImageH"], self.dictModelPara["intImageC"])
            listLayerImage = fn([arrayPhoto, 0]) # get the output of that layer list (1, 1, 48, 48, 64)
            
            fig = plt.figure(figsize=(16, 17))
            intFilters = 16
            for i in range(intFilters):
                ax = fig.add_subplot(intFilters/4, 4, i+1)
                ax.imshow(listLayerImage[0][0, :, :, i], cmap="Blues")
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel("filter {}".format(i))
                plt.tight_layout()
            fig.suptitle("Output of {} (Given image{})".format(listLayerNames[cnt], intChooseId))
            plt.savefig(os.path.join(self.strSaveFolderPath, "FiltersResultImage" + str(intChooseId) + listLayerNames[cnt]))

    def plotSaliencyMap(self, arrayX, arrayYLabel):
        """
        This function plots the saliency map.
        """
        listClasses = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        model = load_model(os.path.join(self.strSaveFolderPath, "model.h5"))

        inputImage = model.input
        listImageIDs = [23, 2, 16, 7, 3, 8, 21, 6, 10, 11]
        for idx in listImageIDs:
            arrayProbability = model.predict(arrayX[idx].reshape(-1, self.dictModelPara["intImageW"], self.dictModelPara["intImageH"], self.dictModelPara["intImageC"]))
            arrayPredictLabel = arrayProbability.argmax(axis=-1)
            tensorTarget = model.output[:, arrayPredictLabel] # ??
            tensorGradients = K.gradients(tensorTarget, inputImage)[0]
            fn = K.function([inputImage, K.learning_phase()], [tensorGradients])

            ### start heatmap processing ###
            arrayGradients = fn([arrayX[idx].reshape(-1, self.dictModelPara["intImageW"], self.dictModelPara["intImageH"], self.dictModelPara["intImageC"]), 0])[0].reshape(self.dictModelPara["intImageW"], self.dictModelPara["intImageH"], -1)
        
            arrayGradients = arrayGradients - arrayGradients.min()
            arrayGradients = arrayGradients / (arrayGradients.max() - arrayGradients.min())

            arrayHeatMap = arrayGradients.reshape(self.dictModelPara["intImageW"], self.dictModelPara["intImageH"])
            ### End heatmap processing ###
            
            print("ID: {}, Truth: {}, Prediction: {}, Confidence:{}".format(idx, arrayYLabel[idx], arrayPredictLabel, arrayProbability[0][int(arrayPredictLabel)]))
            
            # show original image
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(1, 3, 1)
            axx = ax.imshow((arrayX[idx]*255).reshape(self.dictModelPara["intImageW"], self.dictModelPara["intImageH"]), cmap="gray")
            plt.tight_layout()
            
            # show Heat Map
            ax = fig.add_subplot(1, 3, 2)
            axx = ax.imshow(arrayHeatMap, cmap=plt.cm.jet)
            plt.colorbar(axx)
            plt.tight_layout()

            # show Saliency Map
            floatThreshold = 0.55
            arraySee = (arrayX[idx]*255).reshape(self.dictModelPara["intImageW"], self.dictModelPara["intImageH"])
            arraySee[np.where(arrayHeatMap <= floatThreshold)] = np.mean(arraySee)

            ax = fig.add_subplot(1, 3, 3)
            axx = ax.imshow(arraySee, cmap="gray")
            plt.colorbar(axx)
            plt.tight_layout()
            fig.suptitle("Number {}".format(listClasses[listImageIDs.index(idx)]))
            plt.savefig(os.path.join(self.strSaveFolderPath, "SaliencyMap" + listClasses[listImageIDs.index(idx)]))
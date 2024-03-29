from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam


def CNN(dictModelPara):

    inputs = Input(shape=(dictModelPara["intImageW"], dictModelPara["intImageH"], dictModelPara["intImageC"]))

    x = Conv2D(filters=dictModelPara["intConvL1"], kernel_size=dictModelPara["intKernelSize1"], padding=dictModelPara["strPaddingType1"])(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters=dictModelPara["intConvL1"], kernel_size=dictModelPara["intKernelSize1"], padding=dictModelPara["strPaddingType1"])(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=dictModelPara["intPoolSize1"], strides=dictModelPara["intStrides1"])(x)

    x = Conv2D(filters=dictModelPara["intConvL2"], kernel_size=dictModelPara["intKernelSize2"], padding=dictModelPara["strPaddingType2"])(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=dictModelPara["intConvL2"], kernel_size=dictModelPara["intKernelSize2"], padding=dictModelPara["strPaddingType2"])(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=dictModelPara["intPoolSize2"], strides=dictModelPara["intStrides2"])(x)

    x = Flatten()(x)
    x = Dense(dictModelPara["intHiddenSize1"], activation="relu")(x)
    x = Dropout(rate=0.5)(x)
    outputs = Dense(dictModelPara["intClassesNum"], activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    optim = Adam()
    model.compile(loss=["categorical_crossentropy"], optimizer=optim, metrics=["accuracy"])
    model.summary()
    return model


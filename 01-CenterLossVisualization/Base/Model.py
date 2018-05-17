from keras.models import Model
from keras.layers import Input, Conv2D, PReLU, Flatten, Dense, Embedding, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K


def CNN(dictModelPara):

    inputs = Input(shape=(dictModelPara["intImageW"], dictModelPara["intImageH"], dictModelPara["intImageC"]), name="inputs")
    x = Conv2D(filters=dictModelPara["intConvL1"], kernel_size=dictModelPara["intKernelSize1"])(inputs)
    x = PReLU()(x)
    x = Conv2D(filters=dictModelPara["intConvL1"], kernel_size=dictModelPara["intKernelSize1"])(x)
    x = PReLU()(x)
    x = Conv2D(filters=dictModelPara["intConvL2"], kernel_size=dictModelPara["intKernelSize2"])(x)
    x = PReLU()(x)
    x = Conv2D(filters=dictModelPara["intConvL2"], kernel_size=dictModelPara["intKernelSize2"])(x)
    x = PReLU()(x)

    x = Flatten()(x)
    x = Dense(dictModelPara["intHiddenSize"])(x)
    VisualLayer = PReLU(name="VisualLayer")(x)
    outputs = Dense(dictModelPara["intClassesNum"], activation="softmax", name="outputs")(VisualLayer)

    optim = Adam()

    if dictModelPara["boolCenterLoss"]:
        inputLabel = Input(shape=(1,), name="inputLabel")
        centers = Embedding(dictModelPara["intClassesNum"], dictModelPara["intHiddenSize"])(inputLabel)
        l2_loss = Lambda(lambda i: K.sum(K.square(i[0]-i[1][:, 0]), 1, keepdims=True), name="l2_loss")([VisualLayer, centers])
        
        model = Model(inputs=[inputs, inputLabel], outputs=[outputs, l2_loss])
        model.compile(loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred], optimizer=optim, loss_weights=[1., dictModelPara["floatLambda"]], metrics={"outputs":"accuracy"})

    else:
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=["categorical_crossentropy"], optimizer=optim, metrics={"outputs":"accuracy"})

    model.summary()
    return model

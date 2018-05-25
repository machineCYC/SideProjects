import os
import numpy as np
from keras.models import load_model


def makePredict(arrayTest, strOutputFolder):

    strModelPath = os.path.join(strOutputFolder, "02Test/model.h5")
    
    model = load_model(strModelPath)

    predictions = model.predict(arrayTest[0], batch_size=256)

    predictions = np.argmax(predictions, axis=-1)

    return predictions
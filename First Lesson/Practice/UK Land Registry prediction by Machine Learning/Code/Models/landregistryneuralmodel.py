from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.normalization import BatchNormalization

import pdb

class LandRegistryNeuralModel(object):

    @staticmethod
    def build(input_dim, output_size, summary, initMode = 'normal', activation = 'relu', dropoutRate=0.5, neurons1 = 60, neurons2 = 200):
        model = Sequential()
        model.add(Dense(neurons1, kernel_initializer=initMode, input_dim = input_dim))
        model.add(BatchNormalization(axis=1))
        model.add(Activation(activation))
        model.add(Dropout(dropoutRate))

        model.add(Dense(neurons2, kernel_initializer=initMode))
        model.add(Activation(activation))
        model.add(Dropout(dropoutRate))
        model.add(Dense(output_size, kernel_initializer=initMode))

        if summary==True:
            model.summary()

        return model
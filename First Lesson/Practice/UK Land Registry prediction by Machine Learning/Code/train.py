'''
Created on 26/09/2019

@author: Francesco Pugliese
'''

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
import numpy
import os
import timeit
import platform
import argparse
import pdb

# Keras imports
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

# Models imports
from Models.landregistryneuralmodel import LandRegistryNeuralModel

# Other imports
from Preprocessing.preprocessing import LandRegistryPreoprocessing
from Settings.settings import SetParameters
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def create_model(input_dim = 1, output_size = 1, optimizer='sgd', initMode = 'normal', activation = 'relu', neurons1 = 60, neurons2 = 200):
    modelSummary = True
    
    # Build the Land Registry Model 
    neuralLandRegistryNeuralModel = LandRegistryNeuralModel.build(input_dim = input_dim, output_size = output_size, summary = modelSummary, neurons1 = neurons1, neurons2 = neurons2)  
    
    # Compile the Land Registry Model
    neuralLandRegistryNeuralModel.compile(loss="mean_squared_error", optimizer=optimizer)
    
    return neuralLandRegistryNeuralModel


def trainNeuralNet(par):

    # Initialization
    defaultCallbacks = []
    startTime = timeit.default_timer()

    # Preprocess Land Registry Data
    lrp = LandRegistryPreoprocessing(par.demoOnSubset)			 # Test the program on a smaller Dataset - True or Full Dataset - False	
    datasets = lrp.loadLandRegistryData(dataPath=par.datasetPath, dataFile = par.datasetFile, fieldSeparator = par.fieldSeparator, normalizeX=par.normalizeX, normalizeY=par.normalizeY, demoOnSubset = par.demoOnSubset)
    
    trainsetX, trainsetY = datasets[0]
    testsetX, testsetY = datasets[1]
    dataXMax, dataYMax = datasets[2]
    
    # Compute the number of batches per dataset
    nTrainBatches = trainsetX.shape[0] // par.batchSize
    nTestBatches = testsetX.shape[0] // par.batchSize

    print ('\n\nBatch size: %i\n' % par.batchSize)

    print ('Number of training batches: %i' % nTrainBatches)
        
    print ('\nTraining set values size (X): %i x %i' % (trainsetX.shape[0], trainsetX.shape[1]))
    print ('Training set target vector size (Y): %i x 1' % trainsetY.shape[0])
    print ('Sum of train set values (X): %.2f' % trainsetX.sum());
    print ('Sum of train set target (Y): %i' % trainsetY.sum());
    
    print ('Number of test batches: %i' % nTestBatches)
    print ('\nTest set values size (X): %i x %i' % (testsetX.shape[0], testsetX.shape[1]))
    print ('Test set target vector size (Y): %i x 1' % testsetY.shape[0])
    print ('Sum of test set values (X): %.2f' % testsetX.sum());
    print ('Sum of test set target (Y): %i' % testsetY.sum());
	    
    trainIndices = []
    validIndices = []

    # define K-fold cross validation
    if par.crossValidation == True:
        print("\n%d-Fold Cross Validation Activated\n" % (par.K))
        kfold = StratifiedKFold(n_splits=par.K, shuffle=True, random_state=seed)
        
        for trainIndex, validIndex in kfold.split(trainsetX, trainsetY):
            trainIndices.append(trainIndex)                                 # Cros-validation partitioning    
            validIndices.append(validIndex)
    else:
        trainIndices.append(numpy.arange(0, trainsetX.shape[0]))            # Normal partitioning otherwise
        validIndices.append(numpy.arange(0, testsetX.shape[0]))

    for trainIndex, validIndex in zip(trainIndices, validIndices):
        if par.crossValidation == True:
            trainsetX, trainsetY = trainsetX[trainIndex], trainsetY[train_index]
            validsetX, validsetY = trainsetX[validIndex], trainsetY[valid_index]
        else:
            validsetX, validsetY = testsetX, testsetY
       
        # Training Algorithms 
        opt = SGD(lr=par.learningRate)                                 # Stochastic Gradient Descent Training Algorithm

        # CallBacks definition 
        checkPoint=ModelCheckpoint(os.path.join("./",par.modelsPath)+"/"+par.modelFile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        print ('\n')
        
        input_dim = trainsetX.shape[1]
        neuralLandRegistryNeuralModel = create_model(input_dim, par.output_size, opt, 'normal', par.activation, par.neurons1, par.neurons2)                             # Returns the model created                        
        print ('\n\n')

        if par.saveBestModel == True:
            defaultCallbacks = defaultCallbacks+[checkPoint]

        # Fit the Land Registry Model 
        history = neuralLandRegistryNeuralModel.fit(trainsetX, trainsetY, validation_data = (validsetX, validsetY), epochs=par.epochsNumber, batch_size=par.batchSize, callbacks = defaultCallbacks, shuffle = False, verbose=2)	

        print ('\n\nPredicting on %i properties of 2015...\n' % (par.numberOfPropertiesToTest))

        predictions = neuralLandRegistryNeuralModel.predict(testsetX, batch_size = par.batchSize, verbose = 1)   # as the Batch Size might be greater than numberOfPropertiesToTest we test on the whole Test Set
        
        if par.saveBestModel == True:
            bestNeuralLandRegistryNeuralModel = LandRegistryNeuralModel.build(input_dim = input_dim, output_size = par.output_size, summary = False, neurons1 = par.neurons1, neurons2 = par.neurons2)  
            bestNeuralLandRegistryNeuralModel.load_weights(os.path.join("./",par.modelsPath)+"/"+par.modelFile)       
            bestNeuralLandRegistryNeuralModel.compile(loss="mean_squared_error", optimizer=opt)
            bestPredictions = bestNeuralLandRegistryNeuralModel.predict(testsetX, batch_size = par.batchSize, verbose = 1)   # as the Batch Size might be greater than numberOfPropertiesToTest we test on the whole Test Set

    if par.printTestModel == True: 
        logfile = open(os.path.join("./",par.logPath)+"/"+par.logFile, "w")
    
        numpy.set_printoptions(precision = par.logPrecision, suppress = True)
        print ("\n\n\nLast Model: Real Price and Predicted Price : \n") 
        print ("\n\n\nLast Model: Real Price and Predicted Price : \n", file = logfile) 
        
        # Denormalize outputs
        if par.normalizeY == True:
            testsetY = testsetY * dataYMax
            predictions = predictions * dataYMax
            if par.saveBestModel == True:
                bestPredictions = bestPredictions * dataYMax
        
        for i in range(par.numberOfPropertiesToTest):
            print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(predictions[i,], par.logPrecision)))
            print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(predictions[i,], par.logPrecision)), file = logfile)
    
        if par.saveBestModel == True:
            print ("\n\n\nBest Model: Real Price and Predicted Price : \n") 
            print ("\n\n\nBest Model: Real Price and Predicted Price : \n", file = logfile) 
            for i in range(par.numberOfPropertiesToTest):
                print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(bestPredictions[i,], par.logPrecision)))
                print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(bestPredictions[i,], par.logPrecision)), file = logfile)

        logfile.close()
    
    endTime = timeit.default_timer()
    print ('\nTotal time: %.2f minutes' % ((endTime - startTime) / 60.))

if __name__ == '__main__':
    # Operating System
    OS = platform.system()                                      # returns 'Windows', 'Linux', etc
    
    default_config_file = "landregistrymodel.ini"                                                 # Default Configuration File

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", help="configuration file name", required = False)

    (arg1) = parser.parse_args()
    config_file = arg1.conf
    if config_file is None: 
        config_file = default_config_file                                                # Default Configuration File

    ## Configuration of the File Parser
    # Read the Configuration File
    set_parameters = SetParameters("../Conf", config_file, OS) 
    par = set_parameters.read_config_file()

    # Set initial seed for reproducibility
    numpy.random.seed(par.seed) 

    trainNeuralNet(par)

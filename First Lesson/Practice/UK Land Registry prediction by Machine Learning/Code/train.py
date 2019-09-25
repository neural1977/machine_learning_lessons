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

import pdb

# Keras imports
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

# Models imports
from Models.landregistryneuralmodel import LandRegistryNeuralModel

# Other imports
from Preprocessing.preprocessing import LandRegistryPreoprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Operating System
OS = platform.system()                                      # returns 'Windows', 'Linux', etc

# Globals
seed = 23455

# Input
if OS == "Linux":
    datasetPath = '../../../'
elif OS == "Windows": 
    datasetPath = '../Data'

datasetFile = 'pp-complete.csv' 
fieldSeparator = ','                                        # CSV field separator     
demoOnSubset = True											# Test the program on a smaller Dataset

# Output
printTestModel = True
logPath = "../SavedLogs"
logFile = "landregistrylog.txt"
logPrecision = 2
numberOfPropertiesToTest = 10

# Models 
modelsPath = "../SavedModels"
modelFile = "landregistrymodel.dnn"
dropoutRate=0.5
neurons1 = 60 
neurons2 = 200

# Normalization
normalizeX = True                                          # Normalize X between 0 and 1
normalizeY = False                                          # Normalize Y between 0 and 1

# Hyperparameters
epochsNumber = 10
batchSize = 64
learningRate = 0.1		        
output_size = 1                                              # Number of Model's Outputs
saveBestModel = True
earlyStopping = False

# Evaluation
crossValidation = False                                      # K-fold cross validation 
K = 10                                                       # K-fold number

def create_model(input_dim = 1, output_size = 1, optimizer='sgd', initMode = 'normal', activation = 'relu', dropoutRate=0.5, neurons1 = 60, neurons2 = 200):
    modelSummary = True
    
    # Build the Land Registry Model 
    deepLandRegistryNeuralModel = LandRegistryNeuralModel.build(input_dim = input_dim, output_size = output_size, summary = modelSummary, dropoutRate = dropoutRate, neurons1 = neurons1, neurons2 = neurons2)  
    
    # Compile the Land Registry Model
    deepLandRegistryNeuralModel.compile(loss="mean_squared_error", optimizer=optimizer)
    
    return deepLandRegistryNeuralModel


def trainNeuralNet(output_size = output_size):
    # Set initial seed for reproducibility
    numpy.random.seed(seed) 
    
    # Initialization
    defaultCallbacks = []
    startTime = timeit.default_timer()

    # Preprocess Land Registry Data
    lrp = LandRegistryPreoprocessing(demoOnSubset)			 # Test the program on a smaller Dataset - True or Full Dataset - False	
    datasets = lrp.loadLandRegistryData(dataPath=datasetPath, dataFile = datasetFile, fieldSeparator = fieldSeparator, normalizeX=normalizeX, normalizeY=normalizeY, demoOnSubset = demoOnSubset)
    
    trainsetX, trainsetY = datasets[0]
    testsetX, testsetY = datasets[1]
    
    # Compute the number of batches per dataset
    nTrainBatches = trainsetX.shape[0] // batchSize
    nTestBatches = testsetX.shape[0] // batchSize

    print ('\n\nBatch size: %i\n' % batchSize)

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
    if crossValidation == True:
        print("\n%d-Fold Cross Validation Activated\n" % (K))
        kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        
        for trainIndex, validIndex in kfold.split(trainsetX, trainsetY):
            trainIndices.append(trainIndex)                                 # Cros-validation partitioning    
            validIndices.append(validIndex)
    
    else:
        trainIndices.append(numpy.arange(0, trainsetX.shape[0]))            # Normal partitioning otherwise
        validIndices.append(numpy.arange(0, testsetX.shape[0]))

    for trainIndex, validIndex in zip(trainIndices, validIndices):
        if crossValidation == True:
            trainsetX, trainsetY = trainsetX[trainIndex], trainsetY[train_index]
            validsetX, validsetY = trainsetX[validIndex], trainsetY[valid_index]
        else:
            validsetX, validsetY = testsetX, testsetY
       
        # Training Algorithms 
        opt = Adam(lr=learningRate)                                 # Stochastic Gradient Descent Training Algorithm

        # CallBacks definition 
        checkPoint=ModelCheckpoint(os.path.join("./",modelsPath)+"/"+modelFile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        print ('\n')

        input_dim = trainsetX.shape[1]
        deepLandRegistryNeuralModel = create_model(input_dim, output_size, opt)                             # Returns the model created                        

        print ('\n\n')

        if saveBestModel == True:
            defaultCallbacks = defaultCallbacks+[checkPoint]

        # Fit the Land Registry Model 
        history = deepLandRegistryNeuralModel.fit(trainsetX, trainsetY, validation_data = (validsetX, validsetY), epochs=epochsNumber, batch_size=batchSize, callbacks = defaultCallbacks, shuffle = False, verbose=2)	

        print ('\n\nPredicting on %i properties of 2015...\n' % (numberOfPropertiesToTest))

        predictions = deepLandRegistryNeuralModel.predict(testsetX, batch_size = batchSize, verbose = 1)   # as the Batch Size might be greater than numberOfPropertiesToTest we test on the whole Test Set
        if saveBestModel == True:
            bestDeepLandRegistryNeuralModel = LandRegistryNeuralModel.build(input_dim = input_dim, output_size = output_size, summary = False, dropoutRate = dropoutRate, neurons1 = neurons1, neurons2 = neurons2)  
            bestDeepLandRegistryNeuralModel.load_weights(os.path.join("./",modelsPath)+"/"+modelFile)       
            bestDeepLandRegistryNeuralModel.compile(loss="mean_squared_error", optimizer=opt)
            bestPredictions = bestDeepLandRegistryNeuralModel.predict(testsetX, batch_size = batchSize, verbose = 1)   # as the Batch Size might be greater than numberOfPropertiesToTest we test on the whole Test Set

    if printTestModel == True: 
        logfile = open(os.path.join("./",logPath)+"/"+logFile, "w")
    
        numpy.set_printoptions(precision = logPrecision, suppress = True)
        print ("\n\n\nLast Model: Real Price and Predicted Price : \n") 
        print ("\n\n\nLast Model: Real Price and Predicted Price : \n", file = logfile) 
        for i in range(numberOfPropertiesToTest):
            print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(predictions[i,], logPrecision)))
            print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(predictions[i,], logPrecision)), file = logfile)
    
        if saveBestModel == True:
            print ("\n\n\nBest Model: Real Price and Predicted Price : \n") 
            print ("\n\n\nBest Model: Real Price and Predicted Price : \n", file = logfile) 
            for i in range(numberOfPropertiesToTest):
                print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(bestPredictions[i,], logPrecision)))
                print ("%i: RT %i - PT %i" % (i, testsetY[i,], numpy.round(bestPredictions[i,], logPrecision)), file = logfile)

        logfile.close()
    
    endTime = timeit.default_timer()
    print ('\nTotal time: %.2f minutes' % ((endTime - startTime) / 60.))

if __name__ == '__main__':
    trainNeuralNet()

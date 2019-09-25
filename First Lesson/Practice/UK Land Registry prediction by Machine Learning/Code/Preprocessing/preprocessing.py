'''
Created on 26/09/2019

@author: Francesco Pugliese
'''

import numpy as np
import pandas as pd
import os
import pdb

class LandRegistryPreoprocessing(object):
    
    def __init__(self, demoOnSubset):
        self.__demoOnSubset = demoOnSubset
        self.__numberOfRows = 1000000
    
    # Private method extracting only the year info from the second field
    def __yearConverter(self, x):
        year = x.split('-')[0]
        if self.__demoOnSubset == True and year == '1996':									# a trick to evaluate the model on small datasets
            return '2015'
        return year    
	
    # Private method encoding the Lease Duration into F->0, L->1 values    
    def __leaseConverter(self, x):
        if x == 'F':
            return 0
        else: 
            return 1

    # Private method encoding the Location into Other->0, London->1 values
    def __locationConverter(self, x):
        if x=='LONDON': 
            return 1
        else: 
            return 0

    # Private method encoding the Property Type into D = Detached->0, S = Semi-Detached->1, T = Terraced->2, F = Flats/Maisonettes->3, O = Other->4
    def __propertyTypeConverter(self, x):
        if x=='D': 
            return 0
        elif x=='S': 
            return 1
        elif x=='T': 
            return 2
        elif x=='F': 
            return 3
        else:
            return 4

	# Land Registry Data Preprocessing
    def loadLandRegistryData(self, dataPath='', dataFile = '', fieldSeparator = ' ', normalizeX = False, normalizeY = False, demoOnSubset = True):
        # Initialization 
        numberOfRows = None
	
        if demoOnSubset == True:
            numberOfRows = self.__numberOfRows
	
        # Set file
        print('\nLoading Land Registry Dara from the files...\n')
 
        print ('Land Ragistry Data - DataSet Folder : %s' % dataPath)
        print ('Land Ragistry Data - DataSet File : %s\n' % dataFile)

        dataset = os.path.join(dataPath, dataFile)
        if os.path.isfile(dataset):
            # Reads CSV, extracts interesting columns and recode them as input for the model 
            data = np.asarray(pd.read_csv(dataset, sep=fieldSeparator, header=None, nrows = numberOfRows, converters={2:self.__yearConverter, 4:self.__propertyTypeConverter, 6:self.__leaseConverter, 11:self.__locationConverter}, usecols = [1, 2, 4, 6, 11]))             # select only the interested columns
            print ('%i lines read' % (data.shape[0]))
            print ('Number of attributes per line: %i' % (data.shape[1]))
            
            testsetX = data[np.where(data[:,1] == '2015'),:][0][:,2:]                       # Put in TestSet all the data related to 2015, only lease duration, location and property type are considere
            testsetY = data[np.where(data[:,1] == '2015'),:][0][:,0]                        # Test Set Y is properties' price
            trainsetX = data[np.where(data[:,1] <= '2014'),:][0][:,2:]                      # Put in TrainSet all the data excluding those from 2015, only lease duration, location and property type are considere
            trainsetY = data[np.where(data[:,1] <= '2014'),:][0][:,0]                       # Test Set Y is properties' price

            # Normalize the input between 0 - 1
            if normalizeX == True:
                if trainsetX.max()>testsetX.max():                                          # normalize X with respect to overall the data
                    dataXMax = trainsetX.max()   
                else: 
                    dataXMax = testsetX.max() 

                trainsetX = trainsetX / dataXMax                               
                testsetX = testsetX / dataXMax                              
        
            if normalizeY == True:
                if trainsetY.max()>testsetY.max():                                          # normalize Y with respect to overall the data
                    dataYMax = trainsetY.max()   
                else: 
                    dataYMax = testsetY.max() 

                trainsetY = trainsetY / dataYMax
                testsetY = testsetY / dataYMax

            return [(trainsetX, trainsetY), (testsetX, testsetY)]
        else: 
            print("%s - DataSet File does not exist." % (dataset))
		
        return None
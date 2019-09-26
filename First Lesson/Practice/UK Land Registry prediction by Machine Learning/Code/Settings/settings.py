'''
Created on 26/09/2019


@author: Francesco Pugliese
'''

import configparser
import pdb

class SetParameters:
    
    def __init__(self, conf_file_path, conf_file_name, OS):
       # Class Initialization (constructor) 
        self.conf_file_path = conf_file_path
        self.conf_file_name = conf_file_name

        # Globals
        self.seed = 23455
        self.OS = OS

        # Preprocessing
        self.normalizeX = True                                          # Normalize X between 0 and 1
        self.normalizeY = True                                          # Normalize Y between 0 and 1

        # Dataset
        if self.OS == "Linux":
            self.datasetPath = '../../../'
        elif OS == "Windows": 
            self.datasetPath = '../Data'

        self.datasetFile = 'pp-complete.csv' 
        self.fieldSeparator = ','                                        # CSV field separator     
        self.demoOnSubset = False											# Test the program on a smaller Dataset
  
        # Output
        self.printTestModel = True
        self.logPath = "../SavedLogs"
        self.logFile = "landregistrylog.txt"
        self.logPrecision = 2
        self.numberOfPropertiesToTest = 10

        # Models 
        self.modelsPath = '../SavedModels'
        self.modelFile = 'landregistrymodel.dnn'
        self.neurons1 = 2
        self.neurons2 = 5
        self.activation = 'sigmoid'

        # Training
        self.epochsNumber = 2
        self.batchSize = 64
        self.learningRate = 0.1		        
        self.output_size = 1                                              # Number of Model's Outputs
        self.saveBestModel = True
        
        # Evaluation
        self.crossValidation = False                                      # K-fold cross validation 
        self.K = 10                                                       # K-fold number

                
    def read_config_file(self):
        # Read the Configuration File
        config = configparser.ConfigParser()
        config.read(self.conf_file_path+'/'+self.conf_file_name)
        config.sections()

        # Preprocessing
        self.normalizeX = config.getboolean('Preprocessing', 'normalizeX')                                          # Normalize X between 0 and 1
        self.normalizeY = config.getboolean('Preprocessing', 'normalizeY')                                          # Normalize Y between 0 and 1

        # Dataset
        self.datasetFile = config.get('Dataset', 'datasetFile')
        self.fieldSeparator = config.get('Dataset', 'fieldSeparator')               # CSV field separator     
        self.demoOnSubset = config.getboolean('Dataset', 'demoOnSubset')			# Test the program on a smaller Dataset
        
        # Output
        self.printTestModel = config.getboolean('Output', 'printTestModel')
        self.logPath = config.get('Output', 'logPath')
        self.logFile = config.get('Output', 'logFile')
        self.logPrecision = config.getint('Output', 'logPrecision')
        self.numberOfPropertiesToTest = config.getint('Output', 'numberOfPropertiesToTest')

        # Models 
        self.modelsPath = config.get('Models', 'modelsPath')
        self.modelFile = config.get('Models', 'modelFile')
        self.neurons1 = config.getint('Models', 'neurons1')
        self.neurons2 = config.getint('Models', 'neurons2')
        self.activation = config.get('Models', 'activation')
        
        # Training
        self.epochsNumber = config.getint('Training', 'epochsNumber')
        self.batchSize = config.getint('Training', 'batchSize')
        self.learningRate = config.getfloat('Training', 'learningRate')	        
        self.output_size = config.getint('Training', 'output_size')             # Number of Model's Outputs
        self.saveBestModel = config.getboolean('Training', 'saveBestModel')          
        
        # Evaluation
        self.crossValidation = config.getboolean('Evaluation', 'crossValidation')                # K-fold cross validation 
        self.K = config.getint('Evaluation', 'K')                                                                                             # K-fold number

        return self		

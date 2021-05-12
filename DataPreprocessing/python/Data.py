import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Data():
	def  __init__(self, fileName = None):
		self.fileList = []
		self.DataList = []
		self.Data = None
		if fileName != None:
			self.fileList.append(fileName)
		

	def showFiles(self):
		print(self.fileList)

	def addFile(self, fileName):
		if self.checkFile(fileName):
			self.fileList.append(fileName)

	def addFiles(self,fileNames):
		pass

	def checkFile(self,fileName):
		if os.path.isfile(fileName):
			return True
		else:
			logging.error("{} not found".format(fileName))
			return False

	def loadData(self):
		self.DataList = []
		for fileName in self.fileList:
			if ".csv" in fileName:
				self.DataList.append(pd.read_csv(fileName))
		self.Data = pd.concat(self.DataList,axis=0,ignore_index=True)

	# split data into [train,validatio,test]
	def splitData(self, fraction = [0.8,0,0.2], random_seed = 42):
		if len(fraction) != 3 or type(fraction) != type([]):
			logging.error("Length of fraction expected to be 3, splitting fraction should be [train/validation/test]")
			return False
		for factor in fraction:
			if type(factor) != float and type(factor) != int:
				logging.error("Numericla value expected for splitting fraction")
				return False
			elif factor < 0:
				logging.error("Only positive value allowed for splitting fraction")
				return False

		DataCopy = self.Data.copy()
		self.Data_Train = DataCopy.sample(frac = float(fraction[0])/sum(fraction), random_state = random_seed)
		DataNoTrain = DataCopy.drop(self.Data_Train.index)
		self.Data_Validation = DataNoTrain.sample(frac = float(fraction[1])/sum(fraction), random_state = random_seed)
		self.Data_Test = DataNoTrain.drop(self.Data_Validation.index)

	def getTrainData(self):
		return self.Data_Train

	def getValidationData(self):
		return self.Data_Validation

	def getTestData(self):
		return self.Data_Test

	def getTrainDataCopy(self):
		return self.Data_Train.copy()

	def getValidationDataCopy(self):
		return self.Data_Validation.copy()

	def getTestDataCopy(self):
		return self.Data_Test.copy()




		

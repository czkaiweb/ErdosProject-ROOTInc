import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Data():
	def  __init__(self, fileName = None, dataFrame = None):
		self.fileList = []
		self.DataList = []
		self.Data = None
		if fileName != None:
			self.fileList.append(fileName)
		if dataFrame != None and isinstance(df_data, pd.DataFrame):
			self.Data = dataFrame
		
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

	def factorizeData(self):
		factorized_insured = pd.factorize(self.Data["Currently Insured"])
		self.Data["Currently Insured"] = factorized_insured[0]
		factorized_marital = pd.factorize(self.Data["Marital Status"])
		self.Data["Marital Status"] = factorized_marital[0]

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

	def getData(self):
		return self.Data
	
	def getDataCopy(self):
		return self.Data.copy()

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

	# group by feature set, elements in featureSet are (value of feature1, value of feature2, ...), column is a list of column names for these features
	# name is the name of new column storing the  grouping index
	def groupByFeature(self,featureSet=None, columns=None, name="GroupIndex"):
		if featureSet == None:
			print("No feature set to be grouped with")
			return False 
		
		if len(list(featureSet)[0])!=len(columns):
			print("length of features are not consistent with length of column list")
		
		if hasattr(self, 'Data'):
			self.fillGroupIndex(data = self.Data, featureSet = featureSet, columnList = columns, name = name)

		if hasattr(self, 'Data_Train'):
			self.fillGroupIndex(data = self.Data_Train, featureSet = featureSet, columnList = columns, name = name)
		
		if hasattr(self, 'Data_Validation'):
			self.fillGroupIndex(data = self.Data_Validation, featureSet = featureSet, columnList = columns, name = name)

		if hasattr(self, 'Data_Test'):
			self.fillGroupIndex(data = self.Data_Test, featureSet = featureSet, columnList = columns, name = name)

	def fillGroupIndex(self, data = None, featureSet = None, columnList = None, name = "GroupIndex"):
		featureList = [list(elem) for elem in list(featureSet)]
		featureList.sort()

		for i,row in data.iterrows():
			rowfeatureList = []
			for column in columnList:
				rowfeatureList.append(row[column])
			if rowfeatureList in featureList:
				data.at[i,name] = int(featureList.index(rowfeatureList))
			else:
				data.at[i,name] = int(-1)
		

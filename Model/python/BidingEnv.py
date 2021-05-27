import numpy as np
import random

class BidingEnv():
	def __init__(self, initialBudget = 5000, numCustomer = 1000, randomSeed =42,rewardThreshold = 0.1):
		# env_status_space = [customer_status,budget status]
		self.env_status_space = np.array(["",1,1,"",initialBudget])
		self.auction_price_space = np.array([0.0])
		self.next_customer = None
		self.current_customer = None
		self.current_budget = np.array([initialBudget])
		self.initialBudget = np.array([initialBudget])


		# Pool of customer
		self.customerPool = None
		self.sizeOfPool = -1

		# Setup randomlized customer 
		self.randomSeed = randomSeed
		self.rewardThreshold = rewardThreshold

	def seed(self, randomSeed):
		self.randomSeed = randomSeed

	def setRewardThreshold(self,rewardThreshold):
		self.rewardThreshold = rewardThreshold
	
	def initRandomGenerator(self):
		random.seed(self.randomSeed)
		self.sizeOfPool = len(self.customerPool)
		self.next_customer = random.randrange(0,self.sizeOfPool-1,1)

	def loadCustomerPool(self, customerPool):
		self.customerPool = customerPool.to_numpy()
		self.initRandomGenerator()

	def getCustomerInfo(self):
		self.current_customer = self.next_customer
		self.next_customer = random.randrange(0,self.sizeOfPool-1,1)
		customer_info = self.customerPool[self.next_customer]
		customer_status =  customer_info[:4]
		return customer_status

	def getEnv(self):
		customer_status= self.getCustomerInfo()
		#self.env_status_space = np.concatenate((customer_status,self.current_budget/self.initialBudget-0.5),axis=None)
		self.env_status_space = np.concatenate((customer_status,self.current_budget),axis=None)

	def step(self, biding):
		if biding < self.rewardThreshold:
			self.getEnv()
			rewards, action, result = self.getRewards(isBiding = False)

		else:
			self.auction_price_space = np.array([10])
			self.current_budget = self.current_budget-self.auction_price_space
			self.getEnv()
			rewards, action, result = self.getRewards(isBiding = True)

		stop = False
		if self.current_budget <= 0:
			stop = True
		
		info = [action,result]

		return self.env_status_space, rewards, stop, info
	
	def getRewards(self, isBiding):
		customer_info = self.customerPool[self.current_customer]
		# Very naive version: fixed reward based on boolean
		deal = customer_info[-1]
		if (isBiding):
			if bool(deal) == True:
				return 130, True, True
				#return 200
				#return 100*customer_info[1], True, True
			else:
				return -10, True, False
		else:
			if bool(deal) == True:
				return 0, False, True
			else:
				return 0, False, False



	def reset(self):
		self.current_budget = self.initialBudget
		self.initRandomGenerator()
		self.getEnv()

		return self.env_status_space


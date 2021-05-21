import numpy as np
import random

class BidingEnv():
	def __init__(self, initialBudget = 10000, numCustomer = 1000, randomSeed = 42):
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

	def seed(self, randomSeed):
		self.randomSeed = randomSeed
	
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
		self.env_status_space = np.concatenate((customer_status,self.current_budget/self.initialBudget-0.5),axis=None)

	def step(self, bidingPrice):
		if bidingPrice < 0:
			self.getEnv()
			rewards = 0
		else:
			self.auction_price_space = np.array([bidingPrice])
			self.current_budget = self.current_budget-self.auction_price_space
			self.getEnv()
			rewards = self.getRewards()

		stop = False
		if self.current_budget <= 0:
			stop = True
		
		info = {}

		return self.env_status_space, rewards, stop, info
	
	def getRewards(self):
		customer_info = self.customerPool[self.current_customer]
		# Very naive version: fixed reward based on boolean
		deal = customer_info[-1]
		if bool(deal) == True:
			return 100*customer_info[1]
			#return 200
		else:
			return -10


	def reset(self):
		self.current_budget = self.initialBudget
		self.initRandomGenerator()
		self.getEnv()

		return self.env_status_space


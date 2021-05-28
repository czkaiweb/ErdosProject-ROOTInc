import numpy as np
import random

class BidingEnvi():
	def __init__(self, initialBudget, numCustomer = 1000, randomSeed = 42):
		# env_status_space = [customer_status,budget status]
		self.env_status_space = np.array(["",1,1,"",initialBudget])
		self.auction_price_space = np.array([0.0], dtype=np.float32)
		self.next_customer = None
		self.current_customer = None
		self.state = np.array([initialBudget,0],dtype = float)
		#self.current_budget = initialBudget
		#self.initialBudget = initialBudget
		self.current_budget = np.array([initialBudget], dtype=np.float32)
		self.initialBudget = np.array([initialBudget], dtype=np.float32)


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

	def step(self, biding):
		customer_info = self.customerPool[self.current_customer]
		if biding==0:
			self.getEnv()
			rewards, action, result = self.getRewards(isBiding = False)

		elif customer_info[-2]==False:
			self.getEnv()
			rewards, action, result = self.getRewards(isBiding = True)
		else:
			#self.auction_price_space = np.array([bidingPrice])
			self.current_budget = self.current_budget- 10
			self.getEnv()
			rewards, action, result = self.getRewards(isBiding = True)


		if self.current_budget <= 0:
			stop = True
		else:
		 	stop=False

		info = result
		self.state[0]=self.current_budget

		return self.state, rewards, stop, info

	def getRewards(self, isBiding):
		customer_info = self.customerPool[self.current_customer]
		# Very naive version: fixed reward based on boolean
		deal = customer_info[-1]
		click = customer_info[-2]
		if (isBiding):
			if bool(deal) == True:
				return 50, True, True
				#return 200
				#return 100*customer_info[1], True, True
			elif click==True:
				return -10, True, False
			else:
				return 0, False, False
		else:
			if bool(deal) == True:
				return 0, False, True
			else:
				return 0, False, False


	def reset(self):
		self.current_budget = self.initialBudget
		self.initRandomGenerator()
		self.getEnv()

		return self.state


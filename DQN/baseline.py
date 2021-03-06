##This code checks baselines for the total reward, and policies sold for the constant bidding strategy of 10 for every customer

import math, random
import sys

import numpy as np
from collections import deque
from utils import plotLearning

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.autograd import Variable
from BidingEnvtnew import BidingEnvi
sys.path.append('/home/aniket/Documents/erdos/ErdosProject-ROOTInc/Model/python/')
from Data import *
#from DataPreprocessing.python.Data import *


data = Data()
data.addFile("Root_Insurance_data.csv")
data.loadData()
df_data = data.getDataCopy()
env = BidingEnvi(1000)
env.loadCustomerPool(df_data)


total_reward = 0
actionrec = []
spendrec = []
timerec = []
nsold=0
state = env.reset()
for idx in range(10000):
        print(env.state[0])
        next_state, reward, done, result = env.step(10)

        if result==True:
            nsold+=1

        timerec.append(next_state[1])
        spendrec.append(next_state[0])
        state = next_state
        total_reward += reward

        if done:
            break

y = spendrec
x = timerec
plt.plot(y)
plt.xlabel("time")
plt.ylabel("Remaining budget")
plt.show()
print ("total reward is" + str(total_reward), 'remain budget is' + str(env.state[0]), 'nsold is' + str(nsold))

from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
from BidingEnv import BidingEnv
import matplotlib.pyplot as plt
import time

import TrainDDPG_v2 as TrainDDPG

from DataPreprocessing.python.Data  import Data
import buffer



MAX_EPISODES = 50
MAX_STEPS = 1000
MAX_BUFFER = 10000 #1000000
MAX_TOTAL_REWARD = 100000
HIGH_PRICE_CUT = 20
INITIALBUDGET =  10000


# Load & Split data
rootData = Data("/Users/czkaiweb/Research/ErdosBootCamp/Project/ProjectData/Root_Insurance_data.csv")
rootData.loadData()
rootData.factorizeData()
rootData.splitData(fraction=[0.6,0,0.4],random_seed=42)
data_train_copy = rootData.getTrainDataCopy()
data_test_copy = rootData.getTestDataCopy()

if True:
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	data_train_copy[["Currently Insured","Number of Vehicles","Number of Drivers","Marital Status"]] = sc.fit_transform(data_train_copy[["Currently Insured","Number of Vehicles","Number of Drivers","Marital Status"]])
	data_test_copy[["Currently Insured","Number of Vehicles","Number of Drivers","Marital Status"]] = sc.transform(data_test_copy[["Currently Insured","Number of Vehicles","Number of Drivers","Marital Status"]])

	# Category string type variables to numerial:
	#factorized_insured = pd.factorize(data_train_copy["Currently Insured"])
	#data_train_copy["Currently Insured"] = factorized_insured[0]
	#factorized_marital = pd.factorize(data_train_copy["Marital Status"])
	#data_train_copy["Marital Status"] = factorized_marital[0]

print(data_train_copy.head(5))

	#factorized_insured = pd.factorize(data_test_copy["Currently Insured"])
	#data_test_copy["Currently Insured"] = factorized_insured[0]
	#factorized_marital = pd.factorize(data_test_copy["Marital Status"])
	#data_test_copy["Marital Status"] = factorized_marital[0]

print(data_test_copy.head(5))


# Initialized the environment
env = BidingEnv(initialBudget = INITIALBUDGET)
env.loadCustomerPool(data_train_copy)
env_test = BidingEnv(initialBudget = INITIALBUDGET)
env_test.loadCustomerPool(data_test_copy)


# reproducible
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
RANDOMSEED = int(current_time.split(":")[-1])
env.seed(RANDOMSEED)
env_test.seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

s_dim = env.env_status_space.shape[0]
a_dim = env.auction_price_space.shape[0]
a_bound = HIGH_PRICE_CUT

print(' State Dimensions :- ', s_dim)
print(' Action Dimensions :- ', a_dim)
print(' Action Max :- ', a_bound)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = TrainDDPG.Trainer(s_dim, a_dim, ram)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        state = np.float32(observation)
        #print "shape of state is: " + str(state.shape)
        action = trainer.get_exploration_action(state)
        # if _ep%5 == 0:
        #   # validate every 5th episode
        #   action = trainer.get_exploitation_action(state)
        # else:
        #   # get action based on observation, use exploration policy here
        #   action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)

        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        #   continue

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            ram.add(state, action, reward, new_state)

        observation = new_observation

        # perform optimization
        trainer.optimize()
        if done:
            break

    # check memory consumption and clear memory
    gc.collect()
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)

    if _ep%100 == 0:
        print(_ep)
        #trainer.save_models(_ep)


print('Completed episodes')


#test
total_reward = 0
actionrec = []
spendrec = []
timerec = []
env.dailyBudget = 100
state = env.reset()
for idx in range(288):
    #print "count : "+str(idx)
    state = Variable(torch.from_numpy(state.astype(float))).float()
    action = trainer.actor.forward(state)
    actionrec.append(action) #convert to [0,1] range for record
    #print "action : "+str(action)
    next_state, reward, done, _ = env.step(action)
    timerec.append(next_state[1])
    spendrec.append(next_state[0])
    state = next_state
    print("state : "+str(state))
    #print "time : "+str(nz[0][2])+'---'+'budget: '+str(nz[0][3])
    total_reward += reward

#plot pacing signals
y = actionrec
x = [i for i in range(len(actionrec))]
plt.plot(y)
plt.show()
#plot spendings

y = spendrec
x = timerec
plt.plot(y)
plt.show()
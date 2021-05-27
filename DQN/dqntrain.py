#This is a code using deep q-learning (DQN) for an agent that bids upto a certain price.
#The actions are sampled using an epsilon-greedy policy such that epsilon decays exponentially during an iteration.

import math, random
import sys

import numpy as np
from utils import plotLearning
from utils import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.autograd import Variable
from BidingEnvt import BidingEnvi
from Data import *
#from DataPreprocessing.python.Data import *

sys.path.append('/home/aniket/Documents/erdos/ErdosProject-ROOTInc/Model/python/')

USE_CUDA = torch.cuda.is_available()


data = Data()
data.addFile("Root_Insurance_data.csv")
data.loadData()
df_data = data.getDataCopy()
env = BidingEnvi()
env.loadCustomerPool(df_data)

#the range of possible bidding prices from 0 to action_space_size
action_space_size = 20
num_iters = 60
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = num_iters

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states, 12),#128
            nn.ReLU(),
            nn.Linear(12, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            augmentstate = Variable(torch.FloatTensor(np.concatenate([state,[env.initialBudget[0]]])).unsqueeze(0))
            #Q_value corresponding to all actions
            q_value = self.forward(augmentstate)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(action_space_size)

        return action


def compute_td_loss(batch_size):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss



batch_size = 32
gamma = 0.99

losses = []


model = DQN(3, action_space_size)

if USE_CUDA:
    model = model.cuda()

optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer(20000)
scores, eps_history, rembud =[],[],[]

#train
for iters in range(1,num_iters+1):
    print(iters)
    state = env.reset()
    score = 0
    budgetval = 10000.0
    env.intialBudget = budgetval
    epsilon = epsilon_by_frame(iters)
    for i in range(2000):
        action = model.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        if state is not None and next_state is not None and done==False:
            augstate = np.concatenate([state,[env.initialBudget[0]]])
            augnext_state = np.concatenate([next_state,[env.initialBudget[0]]])
            replay_buffer.push(augstate, action, reward, augnext_state, done)

        state = next_state
        score += reward

        if done==True:
            break

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())

    rembud.append(env.state[0])
    scores.append(score)
    eps_history.append(epsilon)
    avg_score = np.mean(scores)
    print('episode ', iters, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'Remaining budget %.2f' % env.state[0])

print('Maximum reward=', max(scores),'at epsilon=', eps_history[scores.index(max(scores))])
x = [i+1 for i in range(num_iters)]
filename = 'rewards_iters.png'
plotLearning(x, scores, eps_history, filename)



#some plotting and other functions relevant to the training

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import math, random

def plotLearning(x, scores, epsilons, filename, epshow, lines=None):
    if epshow==0:
        bl = np.array([1460 for i in range(len(scores))])
        fig=plt.figure()
        #ax=fig.add_subplot(111, label="1")
        ax2=fig.add_subplot(111, label="2")
        ax3 = fig.add_subplot(111, label="3", frame_on=False)
        # ax.plot(x, epsilons, color="C0")
        # ax.set_xlabel("Episode number", color="C0")
        # ax.set_ylabel("Epsilon", color="C0")
        # ax.tick_params(axis='x', colors="C0")
        # ax.tick_params(axis='y', colors="C0")

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

        ax2.scatter(x, running_avg, color="C1")
        #ax2.xaxis.tick_top()
        #ax2.axes.get_xaxis().set_visible(False)
        #ax2.yaxis.tick_right()
        ax2.set_xlabel('Episode', color="C1")
        ax2.set_ylabel('Total reward', color="C1")
        #ax2.tick_params(axis='y', colors="C1")
        #ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors="C1")
        #ymin, ymax = plt.gca().get_ylim()
        ymin, ymax = ax2.axes.get_ylim()

        ax3.plot(x,bl, color="C2")
        ax3.axes.set_xticks([])
        ax3.axes.set_ylim([ymin,ymax])


        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        plt.savefig(filename)

    else:
        bl = np.array([1460 for i in range(len(scores))])
        fig=plt.figure()
        ax=fig.add_subplot(111, label="1")
        ax2=fig.add_subplot(111, label="2", frame_on=False)
        ax3=fig.add_subplot(111, label="3")
        ax.plot(x, epsilons, color="C0")
        ax.set_xlabel("Episode number", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

        ax2.scatter(x, running_avg, color="C1")
        #ax2.xaxis.tick_top()
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.set_xlabel('x label 2', color="C1")
        ax2.set_ylabel('Total reward', color="C1")
        #ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        #ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors="C1")
        ymin, ymax = ax2.axes.get_ylim()

        ax3.plot(x,bl, color="C2")
        #ax3.axes.set_xticks([])
        ax3.axes.set_ylim([ymin,ymax])


        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        plt.savefig(filename)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

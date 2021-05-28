import pandas as pd
from random import random
from sklearn.model_selection import train_test_split,StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

class naive_bidder:

    def __init__(self,
                 tol=0.1,
                 reinforce=False,
                 pricing='fixed',
                 fixed_reward=100,
                 variable_multiplier=[10,10],
                 variable_base=100):
        self.data_dict = {}
        self.tol = tol
        self.reinforce = reinforce
        self.pricing = pricing
        self.fixed_reward = fixed_reward
        self.variable_base_reward = variable_base
        if(len(variable_multiplier) == 2):
            self.variable_multiplier = variable_multiplier
        elif(len(variable_multiplier) == 1):
            self.variable_multiplier = [variable_multiplier,variable_multiplier]
        else:
            raise ValueError('Invalid value for variable cost multiplier.')

    def train(self,train_data,train_price_rank=True):
        set_of_combinations = set()
        for i in train_data.values:
            set_of_combinations.add(tuple(i[:4]))
        self.data_dict = dict.fromkeys(list(set_of_combinations),{
            'click_count':[0,0,0,0,0],
            'policies_sold':[0,0,0,0,0],
            'total':[0,0,0,0,0],
            # 'price_rank_data':[],
            # 'average_bid':[]
        })
        for data in train_data.values:
            rank = data[5] - 1
            self.data_dict[tuple(data[:4])]['total'][rank] += 1
            self.data_dict[tuple(data[:4])]['policies_sold'][rank] += data[-1]
            if(data[6] == True):
                self.data_dict[tuple(data[:4])]['click_count'][rank] += 1
            # if(train_price_rank == True):
            #     self.data_dict[tuple(data[:4])]['price_rank_data'] += [[data[4],data[5]]]


    def reward_calculator(self,input_tuple):
        if(self.pricing == 'variable'):
            return self.variable_base_reward + self.variable_multiplier[0]*(input_tuple[1]-1) + self.variable_multiplier[1]*(input_tuple[2]-1)
        else:
            return self.fixed_reward

    def test(self,test_df):
        rewards_array = []
        test_data = test_df.to_numpy()
        baseline_reward = []
        # baseline_reward = [100 if i[-1] ==  1 else -10 for i in test_data]
        for row in test_data:
            if(row[-1] == 1):
                baseline_reward += [self.reward_calculator(row[:4])]
            elif(row[-2] == 0):
                baseline_reward += [0]
            else:
                baseline_reward += [-10]
            cust_type_data = self.data_dict[tuple(row[:4])]
            listing_rank = row[-3] - 1
            prob_of_success = cust_type_data['policies_sold'][listing_rank]/cust_type_data['total'][listing_rank]
            random_float = random()
            if(random_float <= prob_of_success or abs(random_float - prob_of_success) <= self.tol):
                if(self.reinforce):
                    self.data_dict[tuple(row[:4])]['total'][listing_rank] += 1
                if(row[-1] == 0):
                    if(row[-2] == 1):
                        rewards_array += [-10]
                    else:
                        rewards_array += [0]
                else:
                    rewards_array += [self.reward_calculator(row[:4])]
                    if(self.reinforce):
                        self.data_dict[tuple(row[:4])]['policies_sold'][listing_rank] += 1
            else:
                rewards_array += [0]
        return baseline_reward, rewards_array

if(__name__ == '__main__'):
    root_ins_df = pd.read_csv('../DataExploration/Root_Insurance_data.csv')
    train_data, test_data = train_test_split(root_ins_df,
                                         stratify=root_ins_df['policies_sold'],
                                         test_size=0.33,
                                         random_state=42)
    skf = StratifiedKFold(n_splits=5)

    baseline_profit = []
    profit_array = []
    for train_index, test_index in skf.split(train_data.loc[:,train_data.columns != 'policies_sold'],
                                             train_data['policies_sold']):

        bidder_inst = naive_bidder(tol=0.0,reinforce=True,pricing='variable')
        bidder_inst.train(train_data.iloc[train_index],train_price_rank=False)
        baseline_reward, rewards_array = bidder_inst.test(train_data.iloc[test_index])
        baseline_profit += [sum(baseline_reward)]
        profit_array += [sum(rewards_array)]
        plt.plot(np.cumsum(baseline_reward),'r',label=f'baseline')
        plt.plot(np.cumsum(rewards_array),'g',label='naive bidding')
        # plt.legend()

    plt.show()

    # plt.figure()
    # plt.plot(baseline_profit,'r')
    # plt.plot(profit_array,'g')
    # plt.show()

    # bidder_1.bid()

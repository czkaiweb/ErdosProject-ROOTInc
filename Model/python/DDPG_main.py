import argparse
import os
import time

from  BidingEnv import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

from DataPreprocessing.python.Data import *

#from Model.python.DDPG_model_v2 import DDPG

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

RANDOMSEED = 24             # random seed
HIGH_PRICE_CUT = 20         # cut at high end of biding price

LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 0.99                # reward discount
TAU = 0.001                  # soft replacement
#MEMORY_CAPACITY = 6000     # size of replay buffer
#BATCH_SIZE = 32             # update batchsize

MEMORY_CAPACITY = 1000     # size of replay buffer
BATCH_SIZE = 512        

MAX_EPISODES = 21          # total number of episodes for training
MAX_EP_STEPS = 1000          # total number of steps for each episode
TEST_PER_EPISODES = 2      # test the model per episodes
VAR = 4                     # control exploration

INITIALBUDGET = 10000

PRETRAIN = False

class DDPG(object):
	"""
	DDPG class
	"""
	def __init__(self, a_dim, s_dim, a_bound):
		self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
		self.pointer = 0
		self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

		W_init = tf.random_normal_initializer(mean=0, stddev=1.0)
		b_init = tf.constant_initializer(0.5)

		def get_actor(input_state_shape, name=''):
			"""
			Build actor network
			:param input_state_shape: state
			:param name: name
			:return: act
			"""
			inputs = tf.keras.layers.Input(shape=(input_state_shape,), name='A_input')
			x = tf.keras.layers.Dense(units=8, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='A_l1')(inputs)
			#x = tf.keras.layers.Dense(units=4, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='A_l2')(x)
			x = tf.keras.layers.Dense(units=a_dim, activation=tf.nn.tanh, kernel_initializer=W_init, bias_initializer=b_init, name='A_a')(x)
			x = tf.keras.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            
			return tf.keras.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

		def get_critic(input_state_shape, input_action_shape, name=''):
			"""
			Build critic network
			:param input_state_shape: state
			:param input_action_shape: act
			:param name: name
			:return: Q value Q(s,a)
			"""
			s = tf.keras.layers.Input(shape=(input_state_shape,), name='C_s_input')
			a = tf.keras.layers.Input(shape=(input_action_shape,), name='C_a_input')
			x_s = tf.keras.layers.Dense(units=8, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='C_l1')(s)
			x_s = tf.keras.layers.Dense(units=1, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='C_l2')(x_s)
			x_a = tf.keras.layers.Dense(units=2, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='C_l3')(a)
			x = tf.keras.layers.Concatenate(axis=-1)([x_s, x_a])
			x = tf.keras.layers.Dense(units=1, kernel_initializer=W_init, bias_initializer=b_init, name='C_preout')(x)
			#x = tf.keras.layers.Dense(units=1, kernel_initializer=W_init, bias_initializer=b_init, name='C_out')(x)
			return tf.keras.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

		self.actor = get_actor(s_dim)
		self.critic = get_critic(s_dim, a_dim)
		#self.actor.train()
		#self.critic.train()

		#Assign parameter
		def copy_para(from_model, to_model):
			"""
			Copy parameters for soft updating
			:param from_model: latest model
			:param to_model: target model
			:return: None
			"""
			for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
				j.assign(i)

		#Actor_target
		self.actor_target = get_actor(s_dim, name='_target')
		copy_para(self.actor, self.actor_target)
		#self.actor_target.eval()

		#Critic_target
		self.critic_target = get_critic(s_dim, a_dim, name='_target')
		copy_para(self.critic, self.critic_target)
		#self.critic_target.eval()

		#self.R = tf.keras.layers.Input([None, 1], tf.float32, 'r')

		#EMA weight
		self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

		self.actor_opt = tf.optimizers.Adam(LR_A)
		self.critic_opt = tf.optimizers.Adam(LR_C)


	def ema_update(self):
		# Update EMA with average method
		paras = self.actor.trainable_weights + self.critic.trainable_weights    
		self.ema.apply(paras)                                                   
		for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
			i.assign(self.ema.average(j))                                       

	# choose action
	def choose_action(self, s):
		"""
		Choose action
		:param s: state
		:return: act
		"""
		return self.actor(np.array([s], dtype=np.float32))[0]

	def learn(self):
		"""
		Update parameters
		:return: None
		"""
		indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    
		bt = self.memory[indices, :]                    
		bs = bt[:, :self.s_dim]                         
		ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  
		br = bt[:, -self.s_dim - 1:-self.s_dim]         
		bs_ = bt[:, -self.s_dim:]                       

		# Critic:
		# br + GAMMA * q_
		with tf.GradientTape() as tape:
			a_ = self.actor_target(bs_)
			q_ = self.critic_target([bs_, a_])
			y = br + GAMMA * q_
			q = self.critic([bs, ba])
			td_error = tf.losses.mean_squared_error(y, q)
		c_grads = tape.gradient(td_error, self.critic.trainable_weights)
		self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

		# Actor:
		with tf.GradientTape() as tape:
			a = self.actor(bs)
			q = self.critic([bs, a])
			a_loss = -tf.reduce_mean(q) 
		a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
		self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

		self.ema_update()


	# save s,a,r,s_
	def store_transition(self, s, a, r, s_):
		"""
		Store data in data buffer
		:param s: state
		:param a: act
		:param r: reward
		:param s_: next state
		:return: None
		"""
		s = s.astype(np.float32)
		s_ = s_.astype(np.float32)

		transition = np.hstack((s, a, [r], s_))
		index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
		self.memory[index, :] = transition
		self.pointer += 1

	def save_ckpt(self):
		"""
		save trained weights
		:return: None
		"""
		if not os.path.exists('model'):
			os.makedirs('model')

		#tl.files.save_weights_to_hdf5('model/ddpg_actor.hdf5', self.actor)
		#tl.files.save_weights_to_hdf5('model/ddpg_actor_target.hdf5', self.actor_target)
		#tl.files.save_weights_to_hdf5('model/ddpg_critic.hdf5', self.critic)
		#tl.files.save_weights_to_hdf5('model/ddpg_critic_target.hdf5', self.critic_target)
		self.actor.save('model/ddpg_actor.hdf5')
		self.actor_target.save('model/ddpg_actor_target.hdf5')
		self.critic.save('model/ddpg_critic.hdf5')
		self.critic_target.save('model/ddpg_critic_target.hdf5')

	def load_ckpt(self):
		"""
		load trained weights
		:return: None
		"""
		#tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor.hdf5', self.actor)
		#tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor_target.hdf5', self.actor_target)
		#tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic.hdf5', self.critic)
		#tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_target.hdf5', self.critic_target)
		self.actor = tf.keras.models.load_model('model/ddpg_actor.hdf5')
		self.actor_target = tf.keras.models.load_model('model/ddpg_actor_target.hdf5')
		self.critic = tf.keras.models.load_model('model/ddpg_critic.hdf5')
		self.critic_target = tf.keras.models.load_model('model/ddpg_critic_target.hdf5')

if __name__ == '__main__':

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
	tf.random.set_seed(RANDOMSEED)

	#Status space
	s_dim = env.env_status_space.shape[0]
	a_dim = env.auction_price_space.shape[0]
	a_bound = HIGH_PRICE_CUT

	print('s_dim',s_dim)
	print('a_dim',a_dim)

	#Call DDPG
	ddpg = DDPG(a_dim, s_dim, a_bound)


	#Trainning
	if args.train:  # train
		
		reward_buffer = []      #Buffer for reward of each epoch
		reward_baseline_buffer = []
		reward_god_buffer = []
		t0 = time.time()        #Timer

		# Pre-train
		if PRETRAIN:
			s = env.reset()
			for iPre in range(100000):
				if iPre%10000 == 0:
					print("pre-train: #{}".format(iPre))
				a = ddpg.choose_action(s)      
				a = np.clip(np.random.normal(a, VAR), -20, 20)
				s_, r, done, info = env.step(a)
				ddpg.store_transition(s, a/20, r/100, s_)

				if done:
					s = env.reset()
			ddpg.learn()

		for i in range(MAX_EPISODES):
			t1 = time.time()
			t = time.localtime()
			current_time = time.strftime("%H:%M:%S", t)
			RANDOMSEED = int(current_time.split(":")[-1])
			env.seed(RANDOMSEED)
			s = env.reset()
			ep_reward = 0       #Reward of current EP
			ep_baseline_reward = 0
			for j in range(MAX_EP_STEPS):
				# Add exploration noise
				a = ddpg.choose_action(s)       

				# Apply random exploration to action
				a = np.clip(np.random.normal(a, VAR), -20, 20)

				# Interact with enviroment
				s_, r, done, info = env.step(a)
				# Save s,a,r,s_
				ddpg.store_transition(s, a/20, r/100, s_)

				# fit Q when pool is full
				if ddpg.pointer > MEMORY_CAPACITY:
					ddpg.learn()

				#step forward
				s = s_  
				ep_reward += r
				if j == MAX_EP_STEPS - 1 or done:
					print(
						'\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
							i, MAX_EPISODES, ep_reward,
							time.time() - t1
						)
					)
				plt.show()
				if done:
					break
			# test
			if i and not i % TEST_PER_EPISODES:
				t1 = time.time()
				current_time = time.strftime("%H:%M:%S", time.localtime())
				env_test.seed(int(current_time.split(":")[-1]))
				s = env_test.reset()
				ep_reward = 0
				ep_baseline_reward = 0
				ep_god_reward = 0
				## TF Table:
				#[TP, TN, FP, FN]
				TFTable = [0,0,0,0] 
				budget_baseline = INITIALBUDGET
				for j in range(MAX_EP_STEPS):

					a = ddpg.choose_action(s)  
					s_, r, done, info = env_test.step(a)

					budget_baseline += -10

					if info == [True,True]:
						TFTable[0] += 1
						if budget_baseline > 0:
							ep_baseline_reward += 130
						ep_god_reward += 130
					elif info == [False,True]:
						TFTable[1] += 1
						if budget_baseline > 0:
							ep_baseline_reward += 130
						ep_god_reward += 130
					elif info == [True,False]:
						TFTable[2] += 1
						if budget_baseline > 0:
							ep_baseline_reward += -10
					elif info == [False,False]:
						TFTable[3] += 1
						if budget_baseline > 0:
							ep_baseline_reward += -10

					s = s_
					ep_reward += r
						
					if j == MAX_EP_STEPS - 1 or done:
						print(
							'\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(i, MAX_EPISODES, ep_reward,time.time() - t1)
						)
						TP, TN, FP, FN = TFTable
						precesion = TP/(TP+FP+0.001)
						recall = TP/(TP+TN+0.001)
						bACC = (TP/(TP+FN+0.001)+TN/(TN+FP+0.001))/2
						print("\n Precision: {0:.4f}   | Recall: {1:.4f}   | balanced ACC: {2:4f}".format(precesion,recall,bACC))
						

						reward_buffer.append(ep_reward)
						reward_baseline_buffer.append(ep_baseline_reward)
						reward_god_buffer.append(ep_god_reward)
					if done:
						break

			drawRaw = True
			drawRatio = False
			if reward_buffer and drawRaw:
				plt.ion()
				plt.cla()
				plt.title('DDPG')
				plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
				plt.plot(np.array(range(len(reward_baseline_buffer))) * TEST_PER_EPISODES, reward_baseline_buffer)
				plt.xlabel('episode steps')
				plt.ylabel('normalized state-action value')
				plt.ylim(min(min(reward_buffer),min(reward_baseline_buffer))-100, max(max(reward_buffer),max(reward_baseline_buffer),0)*1.2)
				plt.show()
				plt.pause(0.1)
			if reward_buffer and drawRatio:
				plt.ion()
				plt.cla()
				plt.title('DDPG')
				plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, list(np.array(reward_buffer)/np.array(reward_god_buffer)))  # plot the episode vt
				plt.plot(np.array(range(len(reward_baseline_buffer))) * TEST_PER_EPISODES, list(np.array(reward_baseline_buffer)/np.array(reward_god_buffer)))
				plt.xlabel('episode steps')
				plt.ylabel('normalized state-action value')
				plt.ylim(-1,1.2)
				plt.show()
				plt.pause(0.1)
		plt.ioff()
		plt.show()
		print('\nRunning time: ', time.time() - t0)
		ddpg.save_ckpt()

	# test
	#ddpg.load_ckpt()
	#while True:
	#	s = env.reset()
	#	for i in range(MAX_EP_STEPS):
	#		env.render()
	#		s, r, done, info = env.step(ddpg.choose_action(s))
	#		if done:
	#			break
import gym
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
from keras.models import load_model
import os
import random
import rewardcollector as rc
import argparse
import experienceReplay2
from gym import wrappers

#type 'python keraslander.py -render' to show animation
parser = argparse.ArgumentParser()
parser.add_argument('-render', action='store_true')
parser.add_argument('-record', action='store_true')
args = parser.parse_args()

LEARNING_RATE = 0.0001
SAVEFILE_NAME = 'keraslander2_save.h5'
GAMMA = 0.95

INITIAL_EPSILON = 0.07
FINAL_EPSILON = 0.001
EPSILON_DECAY = 0.99995
ALPHA = 0.1
TRAIN_BATCH_SIZE = 20
REPLAY_MEMORY_SIZE = 10000
WARMUP_EPISODES_BEFORE_LEARNING = 10 
RECORD_EPISODE_INTERVAL = 1
env = gym.make('LunarLander-v2')

OBSERVATION_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

def record_interval(n):
	global RECORD_EPISODE_INTERVAL
	return n% RECORD_EPISODE_INTERVAL ==0

if args.record:				
	env = wrappers.Monitor(env, './tmp/keraslander2', video_callable=record_interval, force=True)

def create_model():
	model = Sequential()
	model.add(Dense(128, activation='relu',input_shape=(OBSERVATION_SPACE,)))
	model.add(Dense(128, activation='relu'))
	# model.add(Dense(64, activation='softplus'))
	# model.add(Dense(32, activation='softplus'))
	model.add(Dense(ACTION_SPACE, activation='linear'))
	adam = Adam(lr=LEARNING_RATE)
	model.compile(loss='mse',optimizer=adam)
	return model


if os.path.isfile('./' + SAVEFILE_NAME):
	model = load_model(SAVEFILE_NAME)
	print "Existing model loaded"
else:
	model = create_model()
	print "New model created"


experience = experienceReplay2.ExperienceReplay(REPLAY_MEMORY_SIZE)
episode_number = 0
episode_reward = 0
rewardcollector = rc.Rewardcollector(1)
epsilon= INITIAL_EPSILON
total_steps = 0
while True:
	s = env.reset()
	d = False
	step_number = 0
	q = None
	#trailingStates = experienceReplay2.trailingStates(1)
	#trailingStates1 = experienceReplay2.trailingStates(1)

	if epsilon >= FINAL_EPSILON:
		epsilon *= EPSILON_DECAY
	

	while d == False:

		#trailingStates.addState(s)
		#stateSnapshot = trailingStates.getTrailingStates()
		s = np.reshape(s, (1,OBSERVATION_SPACE))
		#s = np.reshape(s,(1,1,8))
		

		if random.random() <= epsilon:
			a = env.action_space.sample()
		else:
			q = model.predict(s)
			a = np.argmax(q[0])
			

		s1,r,d,_ = env.step(int(a))

		#trailingStates1.addState(s1)
		#s1Snapshot = trailingStates1.getTrailingStates()
		s1 = np.reshape(s1, (1,OBSERVATION_SPACE))

		experience.storeMemory((s, a, r, d , s1 ))
		
		episode_reward += r
		

		

		if d == True:
			episode_number += 1
			rewardcollector.collectReward(episode_reward)
			episode_reward = 0 

			memorypack = experience.recallMemory(TRAIN_BATCH_SIZE)
			if episode_number > WARMUP_EPISODES_BEFORE_LEARNING and memorypack != False:
			
				for i,mem in enumerate(memorypack):

					xstate, xaction, xreward , xd, xstate1 = mem					
					xq = model.predict(xstate)
					xq[0,xaction] = xreward
					
					if xd == False:
						xstate1 = np.reshape(xstate1,(1,OBSERVATION_SPACE))
						futurereward = np.amax(model.predict(xstate1)[0])
						#xq[0,xaction] = xreward + (GAMMA * futurereward)
						xq[0,xaction] = ((1-ALPHA)*xreward)+ (ALPHA* (GAMMA * futurereward))
			
					model.fit(xstate, xq, epochs=1, verbose=0)

		s = s1

		if args.render:	
			env.render()
			if step_number %30 == 0 : print q

		step_number += 1
		total_steps += 1

	if episode_number % 1 == 0:
		model.save(SAVEFILE_NAME)
		print "Model saved, episode_number %i, average episode reward %i, total steps %i , epsilon %f" % (episode_number, rewardcollector.getAverageReward(), total_steps, epsilon)

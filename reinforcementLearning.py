import gym  
import numpy as np 
from ast import literal_eval
import time

env = gym.make('FrozenLake-v1', render_mode = 'human')  #Make an environment for the AI using openAI gym library

#Building Q-Table
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES,ACTIONS))

EPISODES = 20000 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.83
GAMMA = 0.96

epsilon = 0.9 #start with a 90% of picking a random value

rewards = []

RENDER = False

Q = np.load("ReinforcementSavedSteps.npy")

while True:
    state = env.reset()[0]
    for _ in range(MAX_STEPS):
        action = np.argmax(Q[state,:])
        state, reward, done, truncated, info  = env.step(action)
        if done: 
            break
    break

"""
#Training
for episode in range(EPISODES):
    state = env.reset()[0]
    for _ in range(MAX_STEPS):
        
        #if RENDER:
        #    env.render()
        
        action = None
        #Code to pick action 
        if np.random.uniform(0,1) < epsilon: #Check if a random selected value is less than epsilon
            action = env.action_space.sample() #Take random action
        else: 
            action = np.argmax(Q[state,:]) #Use Q table to pick best action based on current values

        next_state, reward, done, truncated, info  = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])
        
        state = next_state

        if done: 
            rewards.append(reward)
            epsilon -= 0.001
            break #reached goal

np.save("ReinforcementSavedSteps", Q)
"""


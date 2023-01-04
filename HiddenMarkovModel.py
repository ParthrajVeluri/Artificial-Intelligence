"""
Scenario: 
Weather Prediction
States:
- hot day (1)
- cold day (0)

Transitions:
- If hot day, 80%/20% chance for the next day to be hot/cold day respectively
- If cold day, 70%/30% chance for the next day to be cold/hot day respectively 

Observations: 
- On a hot day mean: 15 min: 5 max: 25
- On a cold day mean: 0 min: -5 max: 5
"""

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8,0.2]) #First day has 80%/20% chance of being Cold/Hot
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],[0.2, 0.8]]) #Refer to Transitions
observation_distribution = tfd.Normal(loc = [0.0,15.0], scale = [5.0,10.0]) # loc holds means and scale holds stdev

model = tfd.HiddenMarkovModel(                               #Num steps is the no. of days to predict for. Here its 7       
    initial_distribution = initial_distribution,             #To predict for 1 week
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7) 

mean = model.mean()     #Calculate the probability

with tf.compat.v1.Session():    #Here we need a tf session because "mean" is a partially defined tensor
    print(f"7 Day Probability: {mean.numpy()}")         #To evaluate partially defined tensors, we need tensor sessions
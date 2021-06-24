import numpy as np
import gym
import matplotlib.pyplot as plt
from acrobot_ga import GA


env = gym.make('LunarLander-v2')
observation = env.reset()
done = False
while not done:
    env.render()
    print(np.round(observation, 2))
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
env.close()

from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *

history = 10

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    list_states = list()

    while True:
        array = list()
        list_states.append(rgb2gray(state))
        if len(list_states) >= history:
            array = list_states[-history:]
            array.reverse()
        else:
            array = list_states[:]
            array.reverse()
            for i in range(len(list_states),history):
                array.append(list_states[0])

        array = np.array(array)
        #print(array.shape)
        array = array.reshape((1,96,96,history))
        #print(array.shape)
        a = agent.predict(array)
        #print(a)

        a = a.reshape(3)

        next_state, r, done, info = env.step(a)
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    agent = Model()
    agent = agent.load_model()
    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    print(results)
    fname = os.path.join("./results", "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

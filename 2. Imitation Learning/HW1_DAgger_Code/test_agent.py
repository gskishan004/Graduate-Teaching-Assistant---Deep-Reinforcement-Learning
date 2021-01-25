from __future__ import print_function

from datetime import datetime
from pyglet.window import key
import numpy as np
import gym
import os
import json
import time
import gzip
import pickle
import argparse

from model import Model
from utils import *

history = 1
b  = True
a = np.array([0.0, 0.0, 0.0]).astype('float32')
def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.2

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)

def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')

def run_episode(env, agent, a, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    list_states = list()
    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        #    state = ...
        
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # a = ...
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
        #array = rgb2gray(array)
        array = array.reshape((1,96,96,history))
        #print(array.shape)
        #print(array.shape())
        '''
        #for continous---
        '''
        print(b, a)
        #if not(b) and all(a == [0.0, 0.0, 0.0]):
        act = agent.predict(array)
        #print(a)
        
        #for discrete---
        #act  = np.array([0.0, 0.0, 0.0]).astype('float32')
        val = np.argmax(act)
        act = np.zeros(3)
        if val == 1:
            act[0] = -1.0
        if val == 2:
            act[0] = 1.0
        if val == 3:
            act[1] = 1.0
        
        #a = a.reshape(3)
        #a = act[:]
            #next_state, r, done, info = env.step(a)  
        if all(a == [0.0, 0.0, 0.0]):
            next_state, r, done, info = env.step(act) 
            samples["action"].append(np.array(act))
        else:
            next_state, r, done, info = env.step(a)
            samples["action"].append(np.array(a))
        #a = np.array([0.0, 0.0, 0.0]).astype('float32')
        episode_reward += r   
        samples["state"].append(state)            # state has shape (96, 96, 3)
             # action has shape (1, 3)
        samples["next_state"].append(next_state)
        samples["reward"].append(r)
        samples["terminal"].append(done)    
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break
        #time.sleep(0.1)

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 5                  # number of episodes to test

    # TODO: load agent
    # agent = Model(...)
    # agent.load("models/agent.ckpt")
    agent = Model()
    agent = agent.load_model()
    env = gym.make('CarRacing-v0').unwrapped
    b = True
    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }
    env.reset()
    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, a, rendering=rendering)
        episode_rewards.append(episode_reward)

    print('... saving data')
    store_data(samples, "./data")
    save_results(episode_rewards, "./results")
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

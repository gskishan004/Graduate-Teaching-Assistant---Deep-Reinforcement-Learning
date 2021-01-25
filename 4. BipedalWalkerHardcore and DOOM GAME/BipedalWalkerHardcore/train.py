import threading
import random
import tensorflow as tf
import numpy as np

from params import train_params
from utils import PrioritizedReplayBuffer
from utils import GaussianNoiseGenerator
from agent import Agent
from learner import Learner
    
tf.reset_default_graph()

np.random.seed(train_params.RANDOM_SEED)
random.seed(train_params.RANDOM_SEED)
tf.set_random_seed(train_params.RANDOM_SEED)

PER_memory = PrioritizedReplayBuffer(train_params.REPLAY_MEM_SIZE, train_params.PRIORITY_ALPHA)
gaussian_noise = GaussianNoiseGenerator(train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.NOISE_SCALE)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

threads = []
run_agent_event = threading.Event()
stop_agent_event = threading.Event()

learner = Learner(sess, PER_memory, run_agent_event, stop_agent_event)
learner.build_network()
learner.build_update_ops()
learner.initialise_vars()
learner_policy_params = learner.actor_net.network_params + learner.actor_net.bn_params

threads.append(threading.Thread(target=learner.run))

for n_agent in range(train_params.NUM_AGENTS):
    agent = Agent(sess, train_params.ENV, train_params.RANDOM_SEED, n_agent)
    agent.build_network(training=True)
    agent.build_update_op(learner_policy_params)
    threads.append(threading.Thread(target=agent.run, args=(PER_memory, gaussian_noise, run_agent_event,
                                                            stop_agent_event)))

for t in threads:
    t.start()

for t in threads:
    t.join()

sess.close()

            
        
    
    
        
        
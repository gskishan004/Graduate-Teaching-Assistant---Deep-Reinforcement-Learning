import tensorflow as tf
import numpy as np

from params import test_params
from agent import Agent


np.random.seed(test_params.RANDOM_SEED)
tf.set_random_seed(test_params.RANDOM_SEED)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

agent = Agent(sess, test_params.ENV, test_params.RANDOM_SEED)
agent.build_network(training=False)

agent.test()

sess.close()

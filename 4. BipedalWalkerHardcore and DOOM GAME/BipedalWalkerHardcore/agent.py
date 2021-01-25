import os
import sys
import tensorflow as tf
import numpy as np
import scipy.stats as ss
from collections import deque

from params import train_params, test_params
from utils import Actor, Actor_BN
from utils import BipedalWalkerWrapper

class Agent:
  
    def __init__(self, sess, env, seed, n_agent=0):
        print("Initialising agent %02d... \n" % n_agent)
         
        self.sess = sess        
        self.n_agent = n_agent
       
        self.env_wrapper = BipedalWalkerWrapper()
        self.env_wrapper.set_random_seed(seed*(n_agent+1))
              
    def build_network(self, training):
        self.state_ph = tf.placeholder(tf.float32, ((None,) + train_params.STATE_DIMS))
        
        if training:
            var_scope = ('actor_agent_%02d'%self.n_agent)
        else:
            var_scope = ('learner_actor_main')
          
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=False, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params + self.actor_net.bn_params
        else:
            self.actor_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params
                        
    def build_update_op(self, learner_policy_params):
        update_op = []
        from_vars = learner_policy_params
        to_vars = self.agent_policy_params
                
        for from_var,to_var in zip(from_vars,to_vars):
            update_op.append(to_var.assign(from_var))
        
        self.update_op = update_op
            
    def run(self, PER_memory, gaussian_noise, run_agent_event, stop_agent_event):

        self.exp_buffer = deque()
        
        self.sess.run(self.update_op)
        
        run_agent_event.set()
        
        num_eps = 0
        rewards = []
        while not stop_agent_event.is_set():
            num_eps += 1
            # Reset environment and experience buffer
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            self.exp_buffer.clear()
            
            num_steps = 0
            episode_reward = 0
            ep_done = False
            print(f"Episode: {num_eps}")
            while not ep_done:
                num_steps += 1
                ## Take action and store experience
                if train_params.RENDER:
                    self.env_wrapper.render()
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                action += (gaussian_noise() * train_params.NOISE_DECAY**num_eps)
                next_state, reward, terminal = self.env_wrapper.step(action)
                
                episode_reward += reward 
                               
                next_state = self.env_wrapper.normalise_state(next_state)
                reward = self.env_wrapper.normalise_reward(reward)
                
                self.exp_buffer.append((state, action, reward))
                
                # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE
                    
                    # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                    run_agent_event.wait()   
                    PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                
                state = next_state
                
                if terminal or num_steps == train_params.MAX_EP_LENGTH:
                    # Log total episode reward
                    if train_params.LOG_DIR is not None:
                        summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: episode_reward})
                        self.summary_writer.add_summary(summary_str, num_eps)
                    # Compute Bellman rewards and add experiences to replay memory for the last N-1 experiences still remaining in the experience buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = train_params.DISCOUNT_RATE
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= train_params.DISCOUNT_RATE
                        
                        # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                        run_agent_event.wait()     
                        PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                    
                    # Start next episode
                    ep_done = True
                
            # Update agent networks with learner params every 'update_agent_ep' episodes
            rewards.append(episode_reward)
            if num_eps % train_params.UPDATE_AGENT_EP == 0:
                self.sess.run(self.update_op)
        print(rewards)
        self.env_wrapper.close()
    
    def test(self):   

        def load_ckpt(ckpt_dir, ckpt_file):
            loader = tf.train.Saver()
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
             
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
        
        load_ckpt(test_params.CKPT_DIR, test_params.CKPT_FILE)
            
        rewards = [] 

        for test_ep in range(1, test_params.NUM_EPS_TEST+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            ep_reward = 0
            step = 0
            ep_done = False
            
            while not ep_done:
                if test_params.RENDER:
                    self.env_wrapper.render()
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, reward, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalise_state(state)
                
                ep_reward += reward
                step += 1
                 
                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == test_params.MAX_EP_LENGTH:
                    sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d}'.format(test_ep, test_params.NUM_EPS_TEST))
                    sys.stdout.flush()   
                    rewards.append(ep_reward)
                    ep_done = True   
                
        mean_reward = np.mean(rewards)
        error_reward = ss.sem(rewards)
                
        sys.stdout.write('\x1b[2K\rTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
        sys.stdout.flush()
         
        if test_params.RESULTS_DIR is not None:
            if not os.path.exists(test_params.RESULTS_DIR):
                os.makedirs(test_params.RESULTS_DIR)
            output_file = open(test_params.RESULTS_DIR + '/' + test_params.ENV + '.txt' , 'a')
            output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(self.train_ep, mean_reward, error_reward))
            output_file.flush()
            sys.stdout.write('Results saved to file \n\n')
            sys.stdout.flush()      
        
        self.env_wrapper.close()

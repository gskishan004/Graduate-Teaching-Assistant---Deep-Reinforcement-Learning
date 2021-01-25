import gym
import numpy as np
import tensorflow as tf
import random
import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self.neutral_element = neutral_element
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

    def remove_items(self, num_items):
        del self._value[self._capacity:(self._capacity + num_items)]
        neutral_elements = [self.neutral_element for _ in range(num_items)]
        self._value += neutral_elements
        for idx in range(self._capacity - 1, 0, -1):
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):

        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):

        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):

        return super(MinSegmentTree, self).reduce(start, end)

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = (obs_t, action, reward, obs_tp1, done, gamma)

        self._storage.append(data)

        self._next_idx += 1

    def remove(self, num_samples):
        del self._storage[:num_samples]
        self._next_idx = len(self._storage)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, gammas = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, gamma = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            gammas.append(gamma)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            gammas)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self.it_capacity = 1
        while self.it_capacity < size * 2:
            self.it_capacity *= 2

        self._it_sum = SumSegmentTree(self.it_capacity)
        self._it_min = MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"

        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def remove(self, num_samples):
        super().remove(num_samples)
        self._it_sum.remove_items(num_samples)
        self._it_min.remove_items(num_samples)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def conv2d(inputs, kernel_size, filters, stride, activation=None, use_bias=True,
           weight_init=tf.contrib.layers.xavier_initializer(), bias_init=tf.zeros_initializer(), scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation,
                                    use_bias=use_bias, kernel_initializer=weight_init)
        else:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation,
                                    use_bias=use_bias, kernel_initializer=weight_init,
                                    bias_initializer=bias_init)


def batchnorm(inputs, is_training, momentum=0.9, scope='batch_norm'):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(inputs, momentum=momentum, training=is_training, fused=True)


def dense(inputs, output_size, activation=None, weight_init=tf.contrib.layers.xavier_initializer(),
          bias_init=tf.zeros_initializer(), scope='dense'):
    with tf.variable_scope(scope):
        return tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=weight_init,
                               bias_initializer=bias_init)


def flatten(inputs, scope='flatten'):
    with tf.variable_scope(scope):
        return tf.layers.flatten(inputs)


def relu(inputs, scope='relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(inputs)


def tanh(inputs, scope='tanh'):
    with tf.variable_scope(scope):
        return tf.nn.tanh(inputs)


def softmax(inputs, scope='softmax'):
    with tf.variable_scope(scope):
        return tf.nn.softmax(inputs)

class Critic:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms,
                 v_min, v_max, scope='critic'):

        self.state = state
        self.action = action
        self.state_dims = np.prod(
            state_dims)
        self.action_dims = np.prod(action_dims)
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(self.state_dims))), 1 / tf.sqrt(tf.to_float(self.state_dims))),
                                    bias_init=tf.random_uniform_initializer(
                                        (-1 / tf.sqrt(tf.to_float(self.state_dims))),
                                        1 / tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1 = relu(self.dense1_mul, scope='dense1')

            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                 bias_init=tf.random_uniform_initializer(
                                     (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                     1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))), scope='dense2a')

            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                 bias_init=tf.random_uniform_initializer(
                                     (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                     1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))), scope='dense2b')

            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')

            self.output_logits = dense(self.dense2, num_atoms,
                                       weight_init=tf.random_uniform_initializer(-1 * final_layer_init,
                                                                                 final_layer_init),
                                       bias_init=tf.random_uniform_initializer(-1 * final_layer_init, final_layer_init),
                                       scope='output_logits')

            self.output_probs = softmax(self.output_logits, scope='output_probs')

            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = []

            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)

            self.Q_val = tf.reduce_sum(
                self.z_atoms * self.output_probs)

            self.action_grads = tf.gradients(self.output_probs, self.action,
                                             self.z_atoms)

    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)

                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)

                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits,
                                                                    labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)

                self.l2_reg_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss

                train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)

                return train_step


class Actor:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size,
                 final_layer_init, scope='actor'):

        self.state = state
        self.state_dims = np.prod(
            state_dims)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(self.state_dims))), 1 / tf.sqrt(tf.to_float(self.state_dims))),
                                    bias_init=tf.random_uniform_initializer(
                                        (-1 / tf.sqrt(tf.to_float(self.state_dims))),
                                        1 / tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1 = relu(self.dense1_mul, scope='dense1')

            self.dense2_mul = dense(self.dense1, dense2_size,
                                    weight_init=tf.random_uniform_initializer((-1 / tf.sqrt(tf.to_float(dense1_size))),
                                                                              1 / tf.sqrt(tf.to_float(dense1_size))),
                                    bias_init=tf.random_uniform_initializer((-1 / tf.sqrt(tf.to_float(dense1_size))),
                                                                            1 / tf.sqrt(tf.to_float(dense1_size))),
                                    scope='dense2')

            self.dense2 = relu(self.dense2_mul, scope='dense2')

            self.output_mul = dense(self.dense2, self.action_dims,
                                    weight_init=tf.random_uniform_initializer(-1 * final_layer_init, final_layer_init),
                                    bias_init=tf.random_uniform_initializer(-1 * final_layer_init, final_layer_init),
                                    scope='output')

            self.output_tanh = tanh(self.output_mul, scope='output')

            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh,
                                                       (self.action_bound_high - self.action_bound_low)) + (
                                                  self.action_bound_high + self.action_bound_low))

            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = []  # No batch norm params

    def train_step(self, action_grads, learn_rate, batch_size):

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size),
                                             self.grads))

                train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))

                return train_step


class Critic_BN:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms,
                 v_min, v_max, is_training=False, scope='critic'):

        self.state = state
        self.action = action
        self.state_dims = np.prod(
            state_dims)
        self.action_dims = np.prod(action_dims)
        self.is_training = is_training
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')

            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(self.state_dims))), 1 / tf.sqrt(tf.to_float(self.state_dims))),
                                    bias_init=tf.random_uniform_initializer(
                                        (-1 / tf.sqrt(tf.to_float(self.state_dims))),
                                        1 / tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')

            self.dense1 = relu(self.dense1_bn, scope='dense1')

            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                 bias_init=tf.random_uniform_initializer(
                                     (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                     1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))), scope='dense2a')

            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                 bias_init=tf.random_uniform_initializer(
                                     (-1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))),
                                     1 / tf.sqrt(tf.to_float(dense1_size + self.action_dims))), scope='dense2b')

            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')

            self.output_logits = dense(self.dense2, num_atoms,
                                       weight_init=tf.random_uniform_initializer(-1 * final_layer_init,
                                                                                 final_layer_init),
                                       bias_init=tf.random_uniform_initializer(-1 * final_layer_init, final_layer_init),
                                       scope='output_logits')

            self.output_probs = softmax(self.output_logits, scope='output_probs')

            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [v for v in tf.global_variables(scope=self.scope) if
                              'batch_normalization/moving' in v.name]

            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)

            self.Q_val = tf.reduce_sum(
                self.z_atoms * self.output_probs)

            self.action_grads = tf.gradients(self.output_probs, self.action,
                                             self.z_atoms)

    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)

                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)

                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits,
                                                                    labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)

                self.l2_reg_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                               self.scope)  # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)

                return train_step


class Actor_BN:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size,
                 final_layer_init, is_training=False, scope='actor'):

        self.state = state
        self.state_dims = np.prod(
            state_dims)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.is_training = is_training
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')

            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer(
                (-1 / tf.sqrt(tf.to_float(self.state_dims))), 1 / tf.sqrt(tf.to_float(self.state_dims))),
                                    bias_init=tf.random_uniform_initializer(
                                        (-1 / tf.sqrt(tf.to_float(self.state_dims))),
                                        1 / tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')

            self.dense1 = relu(self.dense1_bn, scope='dense1')

            self.dense2_mul = dense(self.dense1, dense2_size,
                                    weight_init=tf.random_uniform_initializer((-1 / tf.sqrt(tf.to_float(dense1_size))),
                                                                              1 / tf.sqrt(tf.to_float(dense1_size))),
                                    bias_init=tf.random_uniform_initializer((-1 / tf.sqrt(tf.to_float(dense1_size))),
                                                                            1 / tf.sqrt(tf.to_float(dense1_size))),
                                    scope='dense2')

            self.dense2_bn = batchnorm(self.dense2_mul, self.is_training, scope='dense2')

            self.dense2 = relu(self.dense2_bn, scope='dense2')

            self.output_mul = dense(self.dense2, self.action_dims,
                                    weight_init=tf.random_uniform_initializer(-1 * final_layer_init, final_layer_init),
                                    bias_init=tf.random_uniform_initializer(-1 * final_layer_init, final_layer_init),
                                    scope='output')

            self.output_tanh = tanh(self.output_mul, scope='output')

            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh,
                                                       (self.action_bound_high - self.action_bound_low)) + (
                                                  self.action_bound_high + self.action_bound_low))

            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [v for v in tf.global_variables(scope=self.scope) if
                              'batch_normalization/moving' in v.name]

    def train_step(self, action_grads, learn_rate, batch_size):

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size),
                                             self.grads))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                               self.scope)
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))

                return train_step


def _l2_project(z_p, p, z_q):
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]

    d_pos = (d_pos - z_q)[None, :, None]
    d_neg = (z_q - d_neg)[None, :, None]
    z_q = z_q[None, :, None]

    d_neg = tf.where(d_neg > 0, 1. / d_neg, tf.zeros_like(d_neg))
    d_pos = tf.where(d_pos > 0, 1. / d_pos, tf.zeros_like(d_pos))

    delta_qp = z_p - z_q
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)

    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)

class GaussianNoiseGenerator:
    def __init__(self, action_dims, action_bound_low, action_bound_high, noise_scale):
        assert np.array_equal(np.abs(action_bound_low), action_bound_high)

        self.action_dims = action_dims
        self.action_bounds = action_bound_high
        self.scale = noise_scale

    def __call__(self):
        noise = np.random.normal(size=self.action_dims) * self.action_bounds * self.scale

        return noise

class EnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def get_state_dims(self):
        return self.env.observation_space.shape

    def get_state_bounds(self):
        return self.env.observation_space.low, self.env.observation_space.high

    def get_action_dims(self):
        return self.env.action_space.shape

    def get_action_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def close(self):
        self.env.close()


class BipedalWalkerWrapper(EnvWrapper):
    def __init__(self):
        EnvWrapper.__init__(self, 'BipedalWalkerHardcore-v3')
        self.v_min = -40.0
        self.v_max = 40.0

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward / 10.0




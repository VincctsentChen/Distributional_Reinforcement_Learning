import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from scipy.stats import beta

# Disable eager execution (disabling eager_execution allows tf2 to run tf1 code)
tf.compat.v1.disable_eager_execution()

class Distributional_RL:
    def __init__(self, sess, model, learning_rate):
        self.learning_rate = learning_rate
        #self.state_size = 4
        self.state_size = 10
        #self.action_size = 2
        self.action_size = 21
        self.model = model
        self.sess = sess
        self.batch_size = 8
        self.gamma = 0.99
        self.quantile_embedding_dim = 128

        self.num_support = 10
        self.V_max = 100  # reward is QALY and assume the maximum life year is 100
        self.V_min = 0   
        self.dz = float(self.V_max - self.V_min) / (self.num_support - 1)
        self.z = [self.V_min + i * self.dz for i in range(self.num_support)] # The supports are: [-5, -5+10/7, -5+20/7,...,5]

        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size]) # feed in the state of the envoronmnet during training
        self.action = tf.compat.v1.placeholder(tf.float32, [None, self.action_size])
        self.dqn_Y = tf.compat.v1.placeholder(tf.float32, [None, 1]) # None corresponds for variable batch sizes. 1 means each element in the batch is a single scalar value.
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.num_support])
        self.M = tf.compat.v1.placeholder(tf.float32, [None, self.num_support])
        self.tau = tf.compat.v1.placeholder(tf.float32, [None, self.num_support])

        self.main_network, self.main_action_support, self.main_params = self._build_network('main') # main network is the primary network for learning and making predictions. It updates frequently.
        self.target_network, self.target_action_support, self.target_params = self._build_network('target') # target network updates less frequently. It provides stable targets for main network to learn

        if self.model == 'C51':
            self.z_space = tf.tile(tf.reshape(self.z, [1, 1, self.num_support]), [self.batch_size, self.action_size, 1]) # reshape the support to size (1,1,8) first, then we transfer the size to (8,2,8)
            self.z_space_with_target_action_support = self.target_action_support * self.z_space # times the support values and probabilities elementwise to obtain the weighted support values
            expand_dim_action = tf.expand_dims(self.action, -1) # -1 means the new dimension will be added at the last position. Now the size is (bacth size, action size, 1)
            self.Q_s_a = self.main_network * expand_dim_action # elementwise multiplication. tf will broadcast the size, so the final dimension would be: (batch size, action size, num_support)
            self.Q_s_a = tf.reduce_sum(self.Q_s_a, axis=1) # aggregate Q values for each sample by actions. So the size changes from (batch size, action size, num_support) to (batch size, num_support)
            self.loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(self.M, tf.math.log(self.Q_s_a + 1e-20)), axis=1))
             # compute the loss between predicted distn and target distn. For example, if we have a1 and a2 for state s,
             # the predicted distn can be \hat{Z}_{s,a1} and  \hat{Z}_{s,a2,}, while the target distn can be Z_{s,a1} and Z_{s,a2}.
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss) # we train the model to minimize the loss

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.compat.v1.assign(v_old, v)) # after a fixed steps, we copy the parameters of main network to target network. Target network updates periodically to produce consistent target values.

    def train(self, memory):
        minibatch = random.sample(memory, self.batch_size) # sample minibatch from replay buffer
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]
        done_stack = [int(i) for i in done_stack]

        if self.model == 'C51':
            Q_next_state = self.sess.run(self.z_space_with_target_action_support, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(np.sum(Q_next_state, axis=2), axis=1)
            prob_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            prob_next_state_action = [prob_next_state[i, action, :] for i, action in enumerate(next_action)]

            m_prob = np.zeros([self.batch_size, self.num_support])

            for i in range(self.batch_size):
                for j in range(self.num_support):
                    Tz = np.fmin(self.V_max, np.fmax(self.V_min, reward_stack[i] + (1 - done_stack[i]) * 0.99 * (self.V_min + j * self.dz)))
                    bj = (Tz - self.V_min) / self.dz

                    lj = np.floor(bj).astype(int)
                    uj = np.ceil(bj).astype(int)

                    blj = bj - lj
                    buj = uj - bj

                    m_prob[i, lj] += (done_stack[i] + (1 - done_stack[i]) * (prob_next_state_action[i][j])) * buj
                    m_prob[i, uj] += (done_stack[i] + (1 - done_stack[i]) * (prob_next_state_action[i][j])) * blj

            m_prob = m_prob / m_prob.sum(axis=1, keepdims=1)

            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.M: m_prob})

    def _build_network(self, name):
        with tf.compat.v1.variable_scope(name):
            if self.model == 'C51':
                layer_1 = tf.compat.v1.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.compat.v1.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.compat.v1.layers.dense(inputs=layer_2, units=self.action_size * self.num_support, activation=None, trainable=True)

                net_pre = tf.reshape(layer_3, [-1, self.action_size, self.num_support])
                net = tf.nn.softmax(net_pre, axis=2)
                net_action = net

        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return net, net_action, params

    def choose_action(self, state):   # use greedy policy to chooce action
        if self.model == 'C51':
            Q = self.sess.run(self.main_action_support, feed_dict={self.state: [state]})   # Q will be a tensor containing the prob distn for each action.
            # The shape of Q is [1, self.action_size, self.num_support].
            z_space = np.repeat(np.expand_dims(self.z, axis=0), self.action_size, axis=0)  # Add the dimension to make the shape be: [action_size, support_size]
            Q_s_a = np.sum(Q[0] * z_space, axis=1)   # Q[0] has shape [action_size, support_size] for prob, and z_space has shape [action_size, support_size] for support.
            # We sum them over by axis=1 to calculate the expected return for each action. So Q_s_a has shape action_size.
            action = np.argmax(Q_s_a)
        return action

# Use the class defined above
memory_size = 10000
memory = deque(maxlen=memory_size)

sess = tf.compat.v1.Session()
#env = gym.make('CartPole-v1')
learning_rate = 0.0001
model = 'C51'
dqn = Distributional_RL(sess, model, learning_rate)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(dqn.assign_ops)

r = tf.compat.v1.placeholder(tf.float32)
rr = tf.compat.v1.summary.scalar('reward', r)
merged = tf.compat.v1.summary.merge_all()
writer = tf.compat.v1.summary.FileWriter('./board/'+model, sess.graph)

episode = 0
reward_list = []
loss_list = []

def discount_reward(reward_list, gamma=0.97):
    total_reward = 0
    for year in range(len(reward_list)):
        total_reward += reward_list[year]*pow(gamma,year)
    
    return total_reward

# generate non_terminal reward from beta distribution
def reward_generator(mu,sigma2 = 0.1):
    # Calculate alpha and beta
    alpha = mu * (mu * (1 - mu) / sigma2 - 1)
    beta_param = (1 - mu) * (mu * (1 - mu) / sigma2 - 1)    
    # Generate a value from the beta distribution
    value = beta.rvs(alpha, beta_param)
    return value

# generate terminal reward from beta distribution
def ter_reward_generator(mu,sigma2 = 0.1,lower_bound = dqn.V_min,upper_bound = dqn.V_max):
    # Calculate alpha and beta
    alpha = mu * (mu * (1 - mu) / sigma2 - 1)
    beta_param = (1 - mu) * (mu * (1 - mu) / sigma2 - 1)    
    # Generate a value from the beta distribution
    value = beta.rvs(alpha, beta_param)
    # Scale and shift the sample to fit the desired range [lower_bound, upper_bound]
    reward = lower_bound + (upper_bound - lower_bound) * value
    return reward


#while True:
while episode < 1000:   # Use 1000 episodes to train the network
    episode += 1
    e = 1. / ((episode / 10) + 1)
    done = False
    #state = env.reset()
    # use one-hot encoding to define states
    state = np.zeros(dqn.state_size)
    state[np.random.choice(range(6))] = 1   # the initial state cannot be terminal state (state 6,7,8,9)
    global_step = 0
    l = 0
    terminal_states = [6,7,8,9]
    year = 0
    reward_episode = []   # record the rewards of the current episodes
    while not done:
        global_step += 1
        if np.random.rand() < e:
            #action = env.action_space.sample()
            action = np.random.choice(range(21))
        else:
            action = dqn.choose_action(state)   # Use the trained network to calculate return expectation of each action, then choose the optimal action

        #next_state, reward, done, _ = env.step(action)    
        # decide the reward based on if the state is terminal state or not
        tran_p = P[state,:,year, action]
        next_state = np.random.choice(state_order,tran_p)
        if next_state in terminal_states:
            done = True
            mean_ter = (rterm[next_state] - dqn.V_min)/(dqn.V_max-dqn.V_min)
            reward = reward_generator(r[state,year,action],sigma2 = 0.1) + ter_reward_generator(mean_ter,sigma2 = 0.1,lower_bound = dqn.V_min,upper_bound = dqn.V_max)
        else:
            reward = reward_generator(r[state,year,action],sigma2 = 0.1)
        reward_episode.append(reward)
        # if done:
        #     reward = -1
        # else:
        #     reward = 0
        print("The length of the memory is: ", len(memory))
        if len(memory) > 1000:   # If len(memory) > 1000, we start to sample minibatch from the replay buffer
            _, loss = dqn.train(memory)
            l += loss
            if global_step % 5 == 0:
                sess.run(dqn.assign_ops)   # Update the target network's weights every 5 steps (stabalize the training so the distribution converges)

        action_one_hot = np.zeros(dqn.action_size)   # transfer action to one hot encoding of length 21
        action_one_hot[action] = 1
        memory.append([state, next_state, action_one_hot, reward, done])
        state = next_state
        year += 1

        if done or global_step==10:   # each episode can have at most 10 transitions (10 years)
        #if done or global_step==200:
            summary = sess.run(merged, feed_dict={r: global_step})
            writer.add_summary(summary, episode)
            total_reward = discount_reward(reward_list, gamma=0.97)
            #print('episode:', episode, 'reward:', global_step, 'expectation loss:', l)
            print('episode:', episode, 'reward:', total_reward, 'expectation loss:', l)
            #reward_list.append(global_step)
            reward_list.append(total_reward)   # record the total reward for the current episode
            loss_list.append(l)
            reward_episode = []   # restore the reward list for the current episode
            break   # if done==True or we reach step size 200 in current episode, we terminate the iteration and move on to the next episode

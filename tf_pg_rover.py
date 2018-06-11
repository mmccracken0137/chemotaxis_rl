'''
Policy-gradient chemotaxis learner
***MUST*** be run with pythonw in order to properly focus the window.
Adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/Policy-Network.ipynb
Also see Karpathy's blog: http://karpathy.github.io/2016/05/31/rl/
ALSO see https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
'''

import pygame
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from chemo_rover import *
from gradient_field import *
#from pg_agent import *

# class pg_agent():
#     def __init__(self, lr, s_size, a_size, h_size):
#         ### establish feed-forward NN.  agent takes state and produces action
#         self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
#         hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
#         self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
#         self.chosen_action = tf.argmax(self.output, 1)
#
#         '''
#         establish training procedure. feed reward and chosen action into the network
#         to compute the loss, and use it to update the network.
#         '''
#         self.reward_holder = tf.placeholder(shape=[None], dtpye=tf.float32)
#         self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
#         self.indices = tf.range(0, tf.shape(self.output)[0]) + tf.shape(self.output)[1] + self.action_holder
#         self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)
#         self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
#
#         tvars = tf.trainable_variables()
#         self.gradient_holders = []
#         for idx,var in enumerate(tvars):
#             placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
#             self.gradient_holders.append(placeholder)
#
#         self.gradients = tf.gradients(self.loss, tvars)
#
#         optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#         self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

'''
Set up a MLP with a single hidden layer.
Input layer: This needs to take in the color value for each of the two sensors on the robot/cell (one at head, one at tail).  Each color is quantified by three (digitized) 8-bit values (0 to 255), so the input layer needs to have 255 * 3 * 2 = 1530 nodes.  Not sure if this will work as a one-hot input; might be able to get away with simply six input values...

We'll give the hidden layer 200 neurons to start (copy Karpathy).

Actions will be simplest possible: each wheel can move forwards (+1), backward (-1), or not at all (0).  For two wheels, the total number of actions is 9.  This forces the wheels to move at only one speed.
'''

# hyperparameters
n_hidden = 200 # number of nodes in the hidden layer
batch_size = 10 # run this many episodes before doing a parameter update
learning_rate = 1e-3
gamma = 0.99 # discount factor
decay_rate = 0.99 # decay factor for RMSprop leaky sum of grad^2 ???
resume = False # resume from previous trainng session?
render = False # run visualization for each episode?
stoch_choice = 0.2 # frequency that non-highest prob action will not be chosen (non-greedy choice)

# model initialization
n_inputs = 255 * 3 * 2
n_actions = 9

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(n_hidden, n_inputs) / np.sqrt(n_inputs) # "Xavier" initialization
    model['W2'] = np.random.randn(n_hidden, n_actions) / np.sqrt(n_hidden) # "Xavier" initialization

grad_buffer = {k : np.zeros_like(v) for k, v in model.items() } #update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    ''' prepro numpy array of two 3-element tuples into a one-hot list '''
    pro_I = np.zeros(n_inputs)
    I = np.flatten(I)
    for i in range(len(I)): # this is so cute, I can't stand it
        pro_I[i*256 + I[i]] = 1
    return pro_I.astye(np.float)

def discount_rewards(r):
    ''' take 1d array of float rewards and compute discounted reward '''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x) # dot input layer with first set of weights
    h[h<0] = 0 # ReLU, baby!!!
    logp = np.dot(model['W2'], h) # dot second set of weights with hidden layer outputs to produce action probs
    p = sigmoid(logp) # map sigmoid onto action probs
    return p, h # return probs array and hidden state

def policy_backward(epx, eph, epdlogp):
    ''' backward pass. (eph is array of intermediate hidden states) '''
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backprop ReLU
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def rover_actions(a_index):
    ''' returns one of the nine rover actions for simplest case. think base 3. '''
    a = divmod(i, 3)
    a = np.array(a)
    a[:] = [x - 1 for x in a]
    return a

xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

# set up some visualization stuff
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (25, 25)
pygame.init()

BLACK = (0,   0,   0  )
WHITE = (255, 255, 255)
GREEN = (0,   255, 0  )
RED   = (255, 0,   0  )
DBLUE = (0,   0,   255)
LBLUE = (245, 245, 255)
DRED  = (255,   0,   0)
LRED  = (255, 245, 245)

### create gradient field
size = (600,300)
border = 15
screen = pygame.display.set_mode((size[0] + 2*border,
                                  size[1] + 2*border))
pygame.display.set_caption("color field")

grad = Grad_field(LRED, DRED, size[0], size[1], 15)
grad.make_pixels()
screen.fill((255,255,255))
pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))

### initialize rover
rover = Rover(int(size[0]/2), int(size[1]/2), np.random.rand() * 360, width = 20, length = 40)

clock = pygame.time.Clock()

max_epis = 2000

for epi in range(max_epis):
    steps = 0
    rover.reset_position(300, 150, np.random.rand() * 360)
    pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))
    running = True

    while running:
        if render:
            clock.tick(60)

        # get rover obs and preprocess
        x = prepro(rover.observation(grad, [rover.h_point, rover.t_point]))

        # forward policy network and sample an action from returned probabilities
        aprob, h = policy_forward(x)
        if np.random.rand() > stoch_choice:
            action = np.argmax(aprob)
        else:
            action = np.random.randint(0, n_actions)

        TKTKTK line 83 of Karpathy

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Flag that we are done so we exit this loop

        # --- Game logic should go here ---> update
        rover.dual_wheel_move(2, 2)
        rover.update()

        # --- Drawing code should go here ---> draw
        if render:
            screen.fill((255,255,255))
            pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))
            rover.rover_draw(screen)
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

        steps += 1

        if np.array_equal([255, 255, 255], rover.observation(grad, [rover.h_point, rover.t_point])[0]):
            running = False
    print(steps)
#Once we have exited the main program loop we can stop the game engine:
pygame.quit()

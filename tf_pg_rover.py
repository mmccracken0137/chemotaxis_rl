'''
Policy-gradient chemotaxis learner
***MUST*** be run with pythonw in order to properly focus the window.
Adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/Policy-Network.ipynb
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

gamma = 0.99
def discount_rewards(r):
    ''' take 1d array of float rewards and compute discounted reward '''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        ### establish feed-forward NN.  agent takes state and produces action
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        '''
        establish training procedure. feed reward and chosen action into the network
        to compute the loss, and use it to update the network.
        '''
        self.reward_holder = tf.placeholder(shape=[None], dtpye=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indices = tf.range(0, tf.shape(self.output)[0]) + tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (25, 25)
pygame.init()
# if sound used
# pygame.mixer.init()

### assets
#sim_dir = os.path.dirname(__file__)

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

max_epis = 50

for epi in range(max_epis):
    steps = 0
    rover.force_position(300, 150, np.random.rand() * 360)
    pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))
    running = True

    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Flag that we are done so we exit this loop

        # --- Game logic should go here ---> update
        #rover.rotate_left(1)
        #all_sprites.update()

        # --- Drawing code should go here ---> draw
        screen.fill((255,255,255))
        pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))
        #rover.dual_wheel_move(2*np.random.rand(), 2*np.random.rand())
        rover.dual_wheel_move(2, 2)
        rover.update()
        rover.rover_draw(screen)
        #print(rover.observation(grad, [rover.h_point, rover.t_point])[0])

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        steps += 1

        if np.array_equal([255, 255, 255], rover.observation(grad, [rover.h_point, rover.t_point])[0]):
            running = False

#Once we have exited the main program loop we can stop the game engine:
pygame.quit()

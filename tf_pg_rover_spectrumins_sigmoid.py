'''
Policy-gradient chemotaxis learner
***MUST*** be run with pythonw in order to properly focus the visualization window.
Adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/Policy-Network.ipynb
Also see Karpathy's blog: http://karpathy.github.io/2016/05/31/rl/
ALSO see https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
'''

import pygame
import numpy as np
import os
import sys
import tensorflow as tf
import pickle

from chemo_rover import *
from gradient_field import *

'''
Set up a MLP with a single hidden layer.
Input layer: This needs to take in the color value for each of the two sensors on the robot/cell (one at head, one at tail).

We'll give the hidden layer 50 neurons to start.

Actions will be simplest possible: each wheel can move forwards (+1), backward (-1), or not at all (0).  For two wheels, the action space has 9 elements.  This forces the wheels to move at only one speed.
'''

# hyperparameters
n_hidden = 20 # number of nodes in the hidden layer
batch_size = 20 # run this many episodes before doing a parameter update
learning_rate = 1e-4
gamma = 0.99 # discount factor
decay_rate = 0.99 # decay factor for RMSprop leaky sum of grad^2 ???
kindness = 0.2 # frequency that non-highest prob action will not be chosen (non-greedy choice)  ALERT ALERT this may not work for multi-output nets (i.e. >2 possible actions)...
random_position_reset = True # reset the position of the rover randomly for each episode

render = False # run visualization for each episode?
resume = False # resume from previous trainng session?
# render = True # run visualization for each episode?
# resume = True # resume from previous trainng session?

write_inits = False # write initial values to file???
save_inits = None
if write_inits:
    save_inits = open('inits.txt', 'w')

write_fom = True # write figure of merit to file.  show training progress...
save_fom = None
if write_fom:
    save_fom = open('fom.txt', 'w')

# number of episodes to run
max_epis =100000
max_steps = 1500

# model initialization
n_inputs = 3 * 2
# n_actions = 2

# if we're going to pick up from a previous training session, load the model. otherwise initialize!
if resume:
    save_file = sys.argv[1]
    print('\nloading model from %s...\n' % save_file)
    model = pickle.load(open(save_file, 'rb'))
else:
    print('\ninitializing fresh model...\n')
    model = {}
    # will try to multiply in the following order: h = W1.x, y = W2.h.  This dictates the dimension of the weights matrices.
    model['W1'] = np.random.randn(n_hidden, n_inputs) / np.sqrt(n_inputs) # Xavier init
    model['W2'] = np.random.randn(n_hidden) / np.sqrt(n_hidden)

grad_buffer = {k : np.zeros_like(v) for k, v in model.items() } #update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    ''' applies softmax to array of values '''
    probs = []
    sum = 0
    max_x = np.amax(x)
    for i in range(len(x)):
        ex = np.exp(x[i] - max_x)
        probs.append(ex)
        sum += ex
    probs = np.array(probs)
    probs /= sum
    return probs

def prepro(I):
    ''' prepro numpy array of two 3-element tuples into a one-hot list '''
    I = np.ndarray.flatten(I)
    I = I / 255.0
    return I.astype(np.float)

def discount_rewards(r):
    ''' take 1d array of float rewards and compute discounted reward '''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x) # multiply W1 by input layer
    h[h<0] = 0 # ReLU, baby!!!
    logp = np.dot(model['W2'], h) # multiply W2 and hidden layer
    p = sigmoid(logp)
    ### TEST
    # p = np.random.rand(n_actions)
    # p = p / p.sum()
    return p, h # return probs array and hidden state

def policy_backward(epx, eph, epdlogp):
    ''' backward pass. (eph is array of intermediate hidden states). see etienne87 '''
    # dW2 = eph.T.dot(epdlogp)
    # dh = epdlogp.dot(model['W2'].T)
    # dh[eph <= 0] = 0 # backprop ReLU
    # dW1 = epx.T.dot(dh)

    # from Karpathy...
    dW2 = np.dot(eph.T, epdlogp).ravel()  # TKTK from stackexchange!!!
    dh = np.outer(epdlogp, model['W2']) # np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backprop ReLU nonlinearity
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def rover_action(a_index):
    '''
    only two possible actions for this script: either wheel moves forward
    '''
    a = np.zeros(2)
    if a_index == 0:
        a[0] = 1
    elif a_index == 1:
        a[1] = 1
    return a

def rover_reward(ob, act, stp):
    ''' defines the reward function for the rover. observation is a list of color tuples. action is a pair of wheel motions. '''
    reward = 0.0
    if ob[0][1] < 10:
        # reward if head is located in very red region... nutrient rich
        reward += 1.0
    if ob[0][1] > 235: # and obs[0][1] < 255:
        reward += -1.0

    # max step cut-off
    if stp > max_steps:
        reward += -1.0
    return reward

def check_running(ob, act, stp, rew):
        '''
        conditions under which to end the episode...
        '''

        run = True
        #end episode if positive or negative reward is achieved
        if rew >= 1 or rew <= -1:
            run = False

        # # end episode if endzones reached
        # if obs[0][1] < 10:
        #     running = False
        # elif obs[0][1] > 235 and obs[0][1] <= 255:
        #     running = False

        #if rover.h_point[0] <= 0 or rover.h_point[0] >= size[0]:
        #    running = False

        # end the episode if either the head or the tail enter the boundary
        # if border != 0:
        #     if np.array_equal([255, 255, 255], rover.observation(grad_field, [rover.h_point, rover.t_point])[0]):
        #         running = False
        #     if np.array_equal([255, 255, 255], rover.observation(grad_field, [rover.h_point, rover.t_point])[1]):
        #         running = False
        return run

# these objs will hold quantities for each episode
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

# set up visualization
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (25, 25) # sets position of window
pygame.init()
size = (200,100)
border = 0
screen = pygame.display.set_mode((size[0] + 2*border,
                                  size[1] + 2*border))
pygame.display.set_caption("color field")
clock = pygame.time.Clock()

# create gradient field
DRED  = (255,   0,   0) # goal color
LRED  = (255, 245, 245) # opposite of goal color
grad_field = Grad_field(LRED, DRED, size[0], size[1], border, wrap_vert=True)
grad_field.make_pixels()
screen.fill((255,255,255))
pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad_field.pixels))

# initialize rover
rover = Rover(int(size[0]/2), int(size[1]/2), np.random.rand() * 360, width = 20, length = 40)

epi = 1

while epi < max_epis:
    epi += 1
    steps = 0
    if random_position_reset:
        rover.reset_position(int(np.random.uniform(0.3, 0.7) * size[0]),
                             0.5 * size[1], np.random.rand() * 360)
    else:
        rover.reset_position(int(size[0] / 2.0),
                             int(size[1] / 2.0), np.random.rand() * 360)

    if write_inits:
        save_inits.write('%f\t%f\t%f\t' % (rover.center[0], rover.center[1], rover.theta))

    init_goal_dist = size[0] - rover.center[0]

    pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad_field.pixels))
    running = True

    while running:
        if render:
            clock.tick(60)

        # get rover obs and preprocess
        x = prepro(rover.observation(grad_field, [rover.h_point, rover.t_point]))
        # print(x)

        # forward policy network and get probabilities for actions
        aprob, h = policy_forward(x)

        a_ind, y = 0, 0
        if np.random.uniform() < aprob:
            a_ind = 0
            y = 1
        else:
            a_ind = 1
            y = 0

        xs.append(x) # observation
        hs.append(h) # hidden state

        dlogps.append(y - aprob)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Flag that we are done so we exit this loop

        # --- Game logic should go here ---> update
        # update and observe, step rover.  Karpathy line 90-ish

        action = rover_action(a_ind)
        rover.dual_wheel_move(action[0], action[1])
        rover.wrap_vertical(grad_field)
        rover.update()

        # --- Drawing code should go here ---> draw
        if render:
            screen.fill((255,255,255))
            pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad_field.pixels))
            rover.rover_draw(screen)
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

        steps += 1

        #
        obs = rover.observation(grad_field, [rover.h_point, rover.t_point])
        reward = rover_reward(obs, action, steps)
        reward_sum += reward
        drs.append(reward)

        running = check_running(obs, action, steps, reward)

        if running == False:
            if write_fom:
                save_fom.write('%d\t%d\t%d\n' % (init_goal_dist, steps, reward))
            if write_inits:
                save_inits.write('%f\t%f\t%f\n' % (rover.center[0], rover.center[1], rover.theta))

            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], [] # reset array memory

            # compute discounted reward
            discounted_epr = discount_rewards(epr)
            # standardize rewards to be unit norm
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # modulate the gradient with the advantage !!!
            epdlogp *= discounted_epr
            # print(epx.shape, eph.shape, epdlogp.shape, discounted_epr.shape)
            grad = policy_backward(epx, eph, epdlogp)
            for k in model:
                #print(grad_buffer[k].shape, grad[k].shape)
                grad_buffer[k] += grad[k] # accumulate grad over batch

            # now we perform the rmsprop parameter update every time a batch is finished
            if epi % batch_size == 0:
                print('\nbatch ended --> back prop!\n')
                for k, v in model.items():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5) # ???
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            # book-keeping
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum * 0.01

            batch_mean = running_reward
            print('episode %d took %d steps.\tepisode reward total was %.2f.\trunning mean: %.4f' % (epi, steps, reward_sum, running_reward))

            # pickle every 100 episodes
            if epi % 100 == 0: pickle.dump(model, open('save_' + str(n_hidden) + 'h.p', 'wb'))

            reward_sum = 0
            obs = None


#Once we have exited the main program loop we can stop the game engine:
pygame.quit()
if write_inits:
    save_inits.close()
if write_fom:
    save_fom.close()

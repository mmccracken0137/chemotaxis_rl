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
n_hidden = 100 # number of nodes in the hidden layer
batch_size = 10 # run this many episodes before doing a parameter update
learning_rate = 1e-3
gamma = 0.99 # discount factor
decay_rate = 0.99 # decay factor for RMSprop leaky sum of grad^2 ???
kindness = 0.2 # frequency that non-highest prob action will not be chosen (non-greedy choice)  ALERT ALERT this may not work for multi-output nets (i.e. >2 possible actions)...
random_position_reset = True # reset the position of the rover randomly for each episode

# render = False # run visualization for each episode?
# resume = False # resume from previous trainng session?
render = True # run visualization for each episode?
resume = False # resume from previous trainng session?


# model initialization
n_inputs = 3 * 2
n_actions = 9

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
    model['W2'] = np.random.randn(n_actions, n_hidden) / np.sqrt(n_hidden)

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
    p = softmax(logp)
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

    dW2 = np.dot(eph.T, epdlogp).T #ravel()  # TKTK from stackexchange!!!
    dh = np.dot(epdlogp, model['W2']) # np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backprop ReLU nonlinearity
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def rover_action(a_index):
    ''' returns one of the nine rover actions for simplest case. think base-3. tens for left wheel, ones for right wheel. '''
    a = divmod(a_index, 3)
    a = np.array(a)
    a[:] = [x - 1 for x in a]
    a = -1.0*a
    return a

def rover_reward(obs, act):
    ''' defines the reward function for the rover. observation is a list of color tuples. action is a pair of wheel motions. '''
    reward = 0.0
    # if obs[0][0] == 255 and obs[0][1] < 10 and obs[0][2] < 10:
    #     # reward if head is located in very red region... nutrient rich
    #     reward += 1
    # elif obs[0][0] == 255 and obs[0][1] < 10 and obs[0][2] < 10:
    #     # penalize if head is located in very pale region... nutirient poor
    #     reward -= 1

    # try summing the six values in the observation.  Dark red equates to a sum near 255 + 255 = 510.  Less good values would be greater than this.
    # sum = np.sum(np.ndarray.flatten(obs))
    # reward += 1e-2 * (2 - sum/550.0) # returns 1 if maximum redness

    #print(obs[0][0], obs[0][1], obs[0][2])
    # reward the head position
    # if obs[0][0] == 255 and obs[0][1] < 40 and obs[0][2] < 40:
    #     # reward if head is located in very red region... nutrient rich
    #     reward += 1
    if obs[0][1] < 50:
        # reward if head is located in very red region... nutrient rich
        reward += 1

    # punish leaving the field
    # if np.array_equal([255, 255, 255], obs[0]) or np.array_equal([255, 255, 255], obs[1]):
    #     reward -= 1

    # # moving violation...
    # penalty = -1e-3
    # reward += penalty * (abs(act[0]) + abs(act[1]))

    return reward

# these objs will hold quantities for each episode
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

# set up visualization
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (25, 25) # sets position of window
pygame.init()
size = (400,300)
border = 15
screen = pygame.display.set_mode((size[0] + 2*border,
                                  size[1] + 2*border))
pygame.display.set_caption("color field")
clock = pygame.time.Clock()

# create gradient field
DRED  = (255,   0,   0)
LRED  = (255, 245, 245)
grad_field = Grad_field(LRED, DRED, size[0], size[1], 15)
grad_field.make_pixels()
screen.fill((255,255,255))
pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad_field.pixels))

# initialize rover
rover = Rover(int(size[0]/2), int(size[1]/2), np.random.rand() * 360, width = 20, length = 40)

# number of episodes to run
max_epis =80000
epi = 0

while epi < max_epis:
    epi += 1
    steps = 0
    if random_position_reset:
        rover.reset_position(int(np.random.uniform(0.2, 0.5) * size[0]),
                             int(np.random.uniform(0.2, 0.8) * size[1]),
                             np.random.rand() * 360)
    else:
        rover.reset_position(int(size[0] / 2.0),
                             int(size[1] / 2.0), np.random.rand() * 360)

    pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad_field.pixels))
    running = True

    while running:
        if render:
            clock.tick(60)

        # get rover obs and preprocess
        x = prepro(rover.observation(grad_field, [rover.h_point, rover.t_point]))
        #print(x)

        # forward policy network and get probabilities for actions
        aprob, h = policy_forward(x)

        # # choose most probable action most of the time, dictated by (1-greed) factor
        # This probably doesn't work for multiple output nodes/actions...
        # if np.random.rand() > kindness:
        #     a_ind = np.argmax(aprob)
        # else:
        #     a_ind = np.random.randint(0, n_actions)

        # This should be more comprehensive than the above... etienne87
        # roll the dice, in the softmax loss
        u = np.random.uniform()
        aprob_cum = np.cumsum(aprob)
        a_ind = np.where(u <= aprob_cum)[0][0]
        #print(u, a, aprob_cum)

        # record intermediates needed for backprop
        xs.append(x) # observation
        hs.append(h) # hidden state
        y = np.zeros_like(aprob) #a_ind
        y[a_ind] = 1 # don't have the "correct" labelfor RL, so substitute the action that we sampled

        # below is modified for multiple actions... see etienne87
        dlogsoftmax = aprob.copy()
        dlogsoftmax[a_ind] -= 1 # discounted reward
        dlogps.append(dlogsoftmax)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Flag that we are done so we exit this loop

        # --- Game logic should go here ---> update
        # update and observe, step rover.  Karpathy line 90-ish

        action = rover_action(a_ind)
        rover.dual_wheel_move(action[0], action[1])
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
        reward = rover_reward(obs, action)
        reward_sum += reward
        drs.append(reward)

        # end the episode if either the head or the tail enter the boundary
        if np.array_equal([255, 255, 255], rover.observation(grad_field, [rover.h_point, rover.t_point])[0]):
            running = False
        if np.array_equal([255, 255, 255], rover.observation(grad_field, [rover.h_point, rover.t_point])[1]):
            running = False

        # end the episode if the number of steps gets too large
        if steps > 5000:
            running = False

        if running == False:
            drs[-1] += -1e-9 # need some very small but non-zero reward/penalty for backprop
            #print('episode %d ended! reward: %f' % (epi, reward_sum))
            # stack together all inputs, hidden states, action grads, and reward for episode

            # print('aprob shape', aprob.shape)
            # print(model['W1'].shape, x.shape, h.shape, model['W2'].shape, aprob.shape, y.shape)
            print(aprob)

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
            for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

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
            print('episode %d took %d steps. episode reward total was %f. running mean: %f' % (epi, steps, reward_sum, running_reward))
            # print('episode %d. episode reward total was %f. running mean: %f' % (epi, reward_sum, running_reward))

            # pickle every 100 episodes
            if epi % 100 == 0: pickle.dump(model, open('save.p', 'wb'))

            reward_sum = 0
            obs = None


#Once we have exited the main program loop we can stop the game engine:
pygame.quit()

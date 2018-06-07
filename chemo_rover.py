'''
Class for chemotaxis rover
'''

import numpy as np
from operator import add, sub
import pygame

class Rover():
    def __init__(self, xinit = 0, yinit = 0, theta = 0, width = 60, length = 80):
        self.width = width
        self.length = length
        self.theta = theta
        self.theta_rad = self.theta*np.pi/180.0
        self.center = [xinit, yinit]
        self.l_point = [self.center[0] + 0.5*self.width*np.cos(self.theta_rad - np.pi/2.0),
                        self.center[1] + 0.5*self.width*np.sin(self.theta_rad - np.pi/2.0)]
        self.r_point = [self.center[0] - 0.5*self.width*np.cos(self.theta_rad - np.pi/2.0),
                        self.center[1] - 0.5*self.width*np.sin(self.theta_rad - np.pi/2.0)]
        self.h_point = [self.center[0] + 0.5*self.length*np.cos(self.theta_rad),
                        self.center[1] + 0.5*self.length*np.sin(self.theta_rad)]
        self.t_point = [self.center[0] - 0.5*self.length*np.cos(self.theta_rad),
                        self.center[1] - 0.5*self.length*np.sin(self.theta_rad)]
        self.center = np.array(self.center)
        self.r_point = np.array(self.r_point)
        self.l_point = np.array(self.l_point)
        self.h_point = np.array(self.h_point)
        self.t_point = np.array(self.t_point)

        ### generate points of body
        self.outline = [self.h_point, self.r_point, self.t_point, self.l_point]
        self.lr_bar_points = [self.l_point, self.r_point]
        self.ht_bar_points = [self.h_point, self.t_point]
        self.h_point = (int(self.h_point[0]), int(self.h_point[1]))

        ### motion parameters
        self.speedx = 0
        self.speedy = 0
        self.theta = 0
        self.last_update = 0.0

    def reset_position(self, xpos = 0, ypos = 0, theta = 0):
        self.theta = theta
        self.theta_rad = self.theta*np.pi/180.0
        self.center = [xpos, ypos]
        self.l_point = [self.center[0] + 0.5*self.width*np.cos(self.theta_rad - np.pi/2.0),
        self.center[1] + 0.5*self.width*np.sin(self.theta_rad - np.pi/2.0)]
        self.r_point = [self.center[0] - 0.5*self.width*np.cos(self.theta_rad - np.pi/2.0),
        self.center[1] - 0.5*self.width*np.sin(self.theta_rad - np.pi/2.0)]
        self.h_point = [self.center[0] + 0.5*self.length*np.cos(self.theta_rad),
        self.center[1] + 0.5*self.length*np.sin(self.theta_rad)]
        self.t_point = [self.center[0] - 0.5*self.length*np.cos(self.theta_rad),
        self.center[1] - 0.5*self.length*np.sin(self.theta_rad)]
        self.center = np.array(self.center)
        self.r_point = np.array(self.r_point)
        self.l_point = np.array(self.l_point)
        self.h_point = np.array(self.h_point)
        self.t_point = np.array(self.t_point)

    def rover_draw(self, screen):
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        self.lr_bar = pygame.draw.lines(screen, BLACK, False,
                                        tuple(self.lr_bar_points), 1)
        self.ht_bar = pygame.draw.lines(screen, BLACK, False,
                                        tuple(self.ht_bar_points), 1)
        self.shell = pygame.draw.lines(screen, BLACK, True,
                                        tuple(self.outline), 1)
        self.head = pygame.draw.circle(screen, GREEN, tuple(self.h_point), 3)

    def translate(self):
        ### manual translate for testing purposes
        self.speedx = 0
        self.speedy = 0
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_LEFT]:
            self.speedx = -2
        if keystate[pygame.K_RIGHT]:
            self.speedx = 2
        if keystate[pygame.K_UP]:
            self.speedy = -2
        if keystate[pygame.K_DOWN]:
            self.speedy = 2

        self.center = [self.center[0] + self.speedx, self.center[1] + self.speedy]
        self.r_point = [self.r_point[0] + self.speedx, self.r_point[1] + self.speedy]
        self.l_point = [self.l_point[0] + self.speedx, self.l_point[1] + self.speedy]
        self.h_point = [self.h_point[0] + self.speedx, self.h_point[1] + self.speedy]
        self.t_point = [self.t_point[0] + self.speedx, self.t_point[1] + self.speedy]

        ### generate points of body
        self.outline = [self.h_point, self.r_point, self.t_point, self.l_point]
        self.lr_bar_points = [self.l_point, self.r_point]
        self.ht_bar_points = [self.h_point, self.t_point]
        self.h_point = (int(self.h_point[0]), int(self.h_point[1]))

    def dual_wheel_move(self, l_speed = 0, r_speed = 0):
        '''
        This is the main motion fucntion for the rover.
        It takes in a speed for each wheel, and moves the l and r
        points assuming a small dt (set by frame rate).
        The center of the rover is then reset as the midpoint of the
        two new l and r points, and the new l and r points are "fixed" to
        make sure that the geometry of the rover stays constant.
        Direction of motion is gotten from h and t points.
        '''

        ### subtract h ant t points to get direction
        direc = np.fromiter(map(sub, self.h_point, self.t_point), dtype=np.float)
        direc = (1.0/np.sqrt(direc[0]**2 + direc[1]**2)) * direc

        ### translate l and r points
        self.l_point = [self.l_point[0] + l_speed * direc[0],
                        self.l_point[1] + l_speed * direc[1]]
        self.r_point = [self.r_point[0] + r_speed * direc[0],
                        self.r_point[1] + r_speed * direc[1]]

        ### set center to be midpoint of new l and r points
        self.center = 0.5 * np.fromiter(map(add, self.l_point, self.r_point), dtype=np.float)

        ### reposition new l and r points so that they are w/2 away from new center
        l_diff = np.fromiter(map(sub, self.l_point, self.center), dtype=np.float)
        l_fac = 0.5 * self.width / np.sqrt(l_diff[0]**2 + l_diff[1]**2)
        self.l_point = l_fac * np.fromiter(map(add, self.center, l_diff), dtype=np.float)

        r_diff = np.fromiter(map(sub, self.r_point, self.center), dtype=np.float)
        r_fac = 0.5 * self.width / np.sqrt(r_diff[0]**2 + r_diff[1]**2)
        self.r_point = r_fac * np.fromiter(map(add, self.center, r_diff), dtype=np.float)

        ### get new direction of lr bar
        perp = np.fromiter(map(sub, self.r_point, self.l_point), dtype=np.float)
        perp /= np.sqrt(perp[0]**2 + perp[1]**2)

        ### get new direction of ht bar
        direc = [-perp[1], perp[0]]
        direc = np.array(direc)

        ### set new h and t points
        self.h_point = np.fromiter(map(add, self.center, self.length/2.0 *direc), dtype=np.float)
        self.t_point = np.fromiter(map(add, self.center, -self.length/2.0 *direc), dtype=np.float)

        ### generate points of body
        self.outline = [self.h_point, self.r_point, self.t_point, self.l_point]
        self.lr_bar_points = [self.l_point, self.r_point]
        self.ht_bar_points = [self.h_point, self.t_point]
        self.h_point = (int(self.h_point[0]), int(self.h_point[1]))

    ### observation returns the gradient values at the points specified
    ### relies on grad_xy_rgb function defined in the gradient class
    def observation(self, gradient, points):
        obs = []
        for p in points:
            obs.append(gradient.grad_xy_rgb(p[0], p[1]))
        obs = np.array(obs)
        obs = obs.astype(int)
        #obs = np.ndarray.flatten(obs)
        return obs

    ### defines the reward for the current state/environment
    def reward(self, gradient):
        return 0

    def update(self):
        self.last_update += pygame.time.get_ticks()

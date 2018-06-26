'''
Demonstration of chemotaxis gradient and rover classes.
'''

import pygame
import numpy as np
import os
import sys

from chemo_rover import *
from gradient_field import *

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
border = 0
screen = pygame.display.set_mode((size[0] + 2*border,
                                  size[1] + 2*border))
pygame.display.set_caption("color field")

grad = Grad_field(LRED, DRED, size[0], size[1], border)
grad.make_pixels()
screen.fill((255,255,255))
pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))

### initialize rover
rover = Rover(int(size[0]/2), int(size[1]/2), np.random.rand() * 2 * np.pi, width = 20, length = 40)
rover.reset_position(300, 150, np.random.rand() * 2 * np.pi)

clock = pygame.time.Clock()

max_epis = 50

for epi in range(max_epis):
    steps = 0
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
        #rover.dual_wheel_move(2*np.random.rand(), 2*np.random.rand())
        #rover.dual_wheel_move(2, 2)
        rover.manual_rotate()
        if border == 0:
            rover.box_norm_vertical(grad)
        elif np.array_equal([255, 255, 255], rover.observation(grad, [rover.h_point, rover.t_point])[0]):
            running = False

        rover.update()

        # --- Drawing code should go here ---> draw
        screen.fill((255,255,255))
        pygame.surfarray.blit_array(screen, pygame.surfarray.map_array(screen, grad.pixels))
        rover.rover_draw(screen)
        #print(rover.observation(grad, [rover.h_point, rover.t_point])[0])

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        steps += 1

    rover.reset_position(300, 150, np.random.rand() * 2 * np.pi)

#Once we have exited the main program loop we can stop the game engine:
pygame.quit()

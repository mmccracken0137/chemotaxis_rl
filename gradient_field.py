'''
Class for generating/using chemotaxis gradients.
'''

import numpy as np

class Grad_field():
    def __init__(self, from_color, to_color, width, height, border=0):
        self.pixels = []
        self.grad_width = width
        self.grad_height = height
        self.border_width = border
        self.total_width = self.grad_width + 2*self.border_width
        self.total_height = self.grad_height + 2*self.border_width
        self.to_color = to_color
        self.from_color = from_color

    def grad_xy_rgb(self, xpos, ypos):
        ### this is eesentially the functional form of the color gradient...
        ### horizontal linear

        if xpos < self.border_width or xpos > self.border_width + self.grad_width:
            r_val, g_val, b_val = 255, 255, 255
        elif ypos < self.border_width or ypos > self.border_width + self.grad_height:
            r_val, g_val, b_val = 255, 255, 255
        else:
            r_val = (self.to_color[0] - self.from_color[0])*(xpos-self.border_width)/self.grad_width + self.from_color[0]
            g_val = (self.to_color[1] - self.from_color[1])*(xpos-self.border_width)/self.grad_width + self.from_color[1]
            b_val = (self.to_color[2] - self.from_color[2])*(xpos-self.border_width)/self.grad_width + self.from_color[2]

        ### gaussian
        # cen = [self.total_width/2, self.total_height/3]
        # g_wid = [100, 60]
        # r_val = from_col[0] + (to_col[0] - from_col[0])*np.exp(-1.0*(xpos - cen[0])**2/2/(g_wid[0]**2)) * np.exp(-1.0*(ypos - cen[1])**2/2/(g_wid[1]**2))
        # g_val = from_col[1] + (to_col[1] - from_col[1])*np.exp(-1.0*(xpos - cen[0])**2/2/(g_wid[0]**2)) *np.exp(-1.0*(ypos - cen[1])**2/2/(g_wid[1]**2))
        # b_val = from_col[2] + (to_col[2] - from_col[2])*np.exp(-1.0*(xpos - cen[0])**2/2/(g_wid[0]**2)) *np.exp(-1.0*(ypos - cen[1])**2/2/(g_wid[1]**2))

        #return((np.round(r_val,2), np.round(g_val,2), np.round(b_val, 2)))
        return((r_val, g_val, b_val))

    def get_rgb(self, xpos, ypos):
        col = (0, 0, 0)
        if xpos < 0 or xpos >= self.total_width or ypos < 0 or ypos >= self.total_height:
            print('error: xy point is off screen:', xpos, ypos)
            sys.exit()
        else:
            col = self.grad_xy_rgb(xpos, ypos)
        return col

    def make_pixels(self):
        channels = []
        for _ in range(3):
            channels.append(np.empty((self.total_height, self.total_width)))
        print('making pixel gradient for dimensions: w', self.total_width, '; h ', self.total_height)

        for i in range(self.total_height):
            for j in range(self.total_width):
                col = self.get_rgb(j, i)
                channels[0][i][j] = int(col[0])
                channels[1][i][j] = int(col[1])
                channels[2][i][j] = int(col[2])

        channels = np.array(channels)
        channels = np.dstack((channels[0], channels[1], channels[2]))
        channels = channels.astype(int)
        channels = np.swapaxes(channels, 0, 1)
        self.pixels = channels

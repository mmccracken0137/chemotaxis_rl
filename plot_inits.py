#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

infil = open(sys.argv[1], 'r')

theta_i = []
x_i, x_f = [], []

line = infil.readline()
arr = line.split(None)

while len(arr) == 6:
    theta_i.append(float(arr[2]))
    x_i.append(float(arr[0]))
    x_f.append(float(arr[3]))

    line = infil.readline()
    arr = line.split(None)

infil.close()

plt.hist(x_i, 50, range=(0,300), alpha=0.5)
plt.hist(x_f, 50, range=(0,300), alpha=0.5)
plt.show()
